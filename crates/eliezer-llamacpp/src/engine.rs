//! The llama.cpp engine. One worker thread per loaded model owns the model, the context and all
//! sequences; everything here blocks, the async side talks to it through channels.
//!
//! The context window of a model is shared by its parallel sequences (`context / parallel` each).
//! Requests are batched: every loop iteration decodes one token of each running sequence (or a
//! chunk of a prompt) in a single `decode` call.

use crate::{
    output::{Format, OutputParser, detect_format},
    template::{ChatTemplate, RenderOptions},
    vision::{extract, has_images},
};
use eliezer_backend::{BackendError, Chunk, Embeddings};
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde_json::Value;
use std::{
    collections::VecDeque,
    num::NonZeroU32,
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};

/// Chunks a sequence may have queued for its client before it is paused (backpressure).
pub(crate) const STREAM_CAPACITY: usize = 256;
/// A sequence only advances while its client has at least this much room.
const MIN_ROOM: usize = 8;
/// Tokens decoded per `decode` call (prompt chunks and generation steps share this budget).
const N_BATCH: u32 = 2048;
/// An image is encoded to at most this many tokens; the image must fit one micro-batch because the
/// attention within an image is not causal.
const IMAGE_MAX_TOKENS: i32 = 2000;
/// Tokens of the micro-batch left to the text around an image.
const IMAGE_MARGIN: u32 = 48;
/// Chat template used when the GGUF has none.
const FALLBACK_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

pub(crate) struct Setup {
    pub key: String,
    pub path: PathBuf,
    pub context: u32,
    pub parallel: u32,
    pub embedding: bool,
    pub gpu_layers: u32,
    /// The vision projector (`mmproj` file) of a model that reads images.
    pub projector: Option<PathBuf>,
}

pub(crate) struct Job {
    pub body: Value,
    pub reasoning: Option<String>,
    pub tx: mpsc::Sender<Result<Chunk, BackendError>>,
    /// Answered once the request is accepted (or refused) so the caller can fail fast.
    pub admitted: oneshot::Sender<Result<(), BackendError>>,
}
pub(crate) struct EmbedJob {
    pub input: Vec<String>,
    pub reply: oneshot::Sender<Result<Embeddings, BackendError>>,
}
pub(crate) enum Cmd {
    Generate(Job),
    Embed(EmbedJob),
    Stop,
}

pub(crate) struct Worker {
    pub cmd: std_mpsc::Sender<Cmd>,
    pub join: std::thread::JoinHandle<()>,
    pub alive: std::sync::Arc<AtomicBool>,
}

/// llama.cpp may be initialised only once per process; the backend lives for the whole process.
fn llama_backend() -> Result<&'static LlamaBackend, BackendError> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if BACKEND.get().is_none() {
        let mut backend = LlamaBackend::init().map_err(|error| {
            tracing::error!(%error, "llama.cpp backend cannot be initialised");
            BackendError::Unavailable
        })?;
        backend.void_logs();
        if !backend.supports_gpu_offload() {
            tracing::warn!("This llama.cpp build has no GPU offload; models run on the CPU");
        }
        let _ = BACKEND.set(backend);
    }
    BACKEND.get().ok_or(BackendError::Unavailable)
}

/// Loads the model on a new thread and returns once it is ready (or failed).
pub(crate) async fn spawn(setup: Setup) -> Result<Worker, BackendError> {
    let (cmd, commands) = std_mpsc::channel();
    let (ready_tx, ready_rx) = oneshot::channel();
    let alive = std::sync::Arc::new(AtomicBool::new(true));
    let flag = alive.clone();
    let join = std::thread::Builder::new()
        .name(format!("llama-{}", setup.key))
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker(&setup, &commands, ready_tx)
            }));
            flag.store(false, Ordering::Release);
            if outcome.is_err() {
                tracing::error!(model = %setup.key, "The llama.cpp worker panicked");
            }
        })
        .map_err(|_| BackendError::Unavailable)?;
    match ready_rx.await {
        Ok(Ok(())) => Ok(Worker { cmd, join, alive }),
        Ok(Err(error)) => {
            let _ = tokio::task::spawn_blocking(move || join.join()).await;
            Err(error)
        }
        Err(_) => Err(BackendError::Crashed),
    }
}

fn worker(
    setup: &Setup,
    commands: &std_mpsc::Receiver<Cmd>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let backend = match llama_backend() {
        Ok(backend) => backend,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let started = std::time::Instant::now();
    let params = LlamaModelParams::default().with_n_gpu_layers(setup.gpu_layers);
    let model = match LlamaModel::load_from_file(backend, &setup.path, &params) {
        Ok(model) => model,
        Err(error) => {
            tracing::error!(model = %setup.key, %error, "The model cannot be loaded");
            let _ = ready.send(Err(BackendError::Unavailable));
            return;
        }
    };
    let slots = if setup.embedding { 1 } else { setup.parallel };
    let Some((mut context, ubatch)) = create_context(&model, backend, setup, slots) else {
        tracing::error!(model = %setup.key, "The context cannot be created");
        let _ = ready.send(Err(BackendError::Unavailable));
        return;
    };
    let mut vision = match &setup.projector {
        None => None,
        Some(path) => match init_vision(path, &model, setup.gpu_layers > 0, ubatch) {
            Ok(ctx) => Some(ctx),
            Err(reason) => {
                tracing::error!(model = %setup.key, reason, "The vision projector cannot be loaded");
                let _ = ready.send(Err(BackendError::Unavailable));
                return;
            }
        },
    };
    tracing::info!(
        event = "model_loaded",
        model = %setup.key,
        context = setup.context,
        parallel = slots,
        millis = started.elapsed().as_millis() as u64,
        "Model loaded"
    );
    let _ = ready.send(Ok(()));
    if setup.embedding {
        embed_loop(&model, &mut context, commands);
    } else {
        chat_loop(&model, &mut context, vision.as_mut(), setup, commands);
    }
    // The projector and the context must go before the model; all release their GPU memory here.
    drop(vision);
    drop(context);
    drop(model);
    tracing::info!(event = "model_unloaded", model = %setup.key, "Model unloaded");
}

/// Creates the context of a model and returns it with the micro-batch size that was reached (0 for
/// the library default).
///
/// Whole inputs (and whole images) must fit one micro-batch for non-causal attention. A big
/// micro-batch needs a big compute buffer, which some models (Gemma 4) cannot get next to their
/// weights, so a model with a projector steps down until its context can be created; the image size
/// follows the micro-batch that was reached.
fn create_context<'m>(
    model: &'m LlamaModel,
    backend: &LlamaBackend,
    setup: &Setup,
    slots: u32,
) -> Option<(LlamaContext<'m>, u32)> {
    let params = |ubatch: Option<u32>| {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(setup.context))
            .with_n_batch(N_BATCH)
            .with_n_seq_max(slots)
            // Sliding-window layers keep only their window (as the memory estimate assumes); the
            // library default keeps the full context for them, which is several GB more for Gemma 4.
            .with_swa_full(false)
            .with_embeddings(setup.embedding);
        match ubatch {
            Some(ubatch) => params.with_n_ubatch(ubatch),
            None => params,
        }
    };
    let ladder: &[Option<u32>] = if setup.projector.is_some() {
        &[Some(N_BATCH), Some(N_BATCH / 2), Some(N_BATCH / 4)]
    } else if setup.embedding {
        &[Some(N_BATCH)]
    } else {
        &[None]
    };
    for ubatch in ladder {
        match model.new_context(backend, params(*ubatch)) {
            Ok(context) => return Some((context, ubatch.unwrap_or(0))),
            Err(error) => tracing::warn!(
                model = %setup.key,
                ubatch = ubatch.unwrap_or(0),
                %error,
                "The context cannot be created"
            ),
        }
    }
    None
}

/// Loads the vision projector next to the model.
fn init_vision(
    path: &std::path::Path,
    model: &LlamaModel,
    use_gpu: bool,
    ubatch: u32,
) -> Result<MtmdContext, &'static str> {
    let path = path.to_str().ok_or("the path is not valid text")?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().div_ceil(2));
    let params = MtmdContextParams {
        use_gpu,
        print_timings: false,
        n_threads: i32::try_from(threads).unwrap_or(4),
        media_marker: std::ffi::CString::new(mtmd_default_marker())
            .map_err(|_| "the media marker is not valid")?,
        image_min_tokens: -1,
        image_max_tokens: i32::try_from(ubatch.saturating_sub(IMAGE_MARGIN))
            .unwrap_or(IMAGE_MAX_TOKENS)
            .min(IMAGE_MAX_TOKENS),
    };
    let context = MtmdContext::init_from_file(path, model, &params)
        .map_err(|_| "llama.cpp refused the projector file")?;
    if !context.support_vision() {
        return Err("the projector does not support images");
    }
    Ok(context)
}

// ---------------------------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------------------------

fn embed_loop(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    commands: &std_mpsc::Receiver<Cmd>,
) {
    while let Ok(cmd) = commands.recv() {
        match cmd {
            Cmd::Stop => break,
            Cmd::Generate(job) => {
                let _ = job.admitted.send(Err(BackendError::InvalidRequest));
            }
            Cmd::Embed(job) => {
                let _ = job.reply.send(embed(model, context, &job.input));
            }
        }
    }
}

fn embed(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    input: &[String],
) -> Result<Embeddings, BackendError> {
    let vocab = model.vocab();
    let limit = context.n_ctx() as usize;
    let mut vectors = Vec::with_capacity(input.len());
    let mut total = 0_u64;
    for text in input {
        let tokens = vocab.tokenize(text.as_bytes(), true, true);
        if tokens.is_empty() || tokens.len() > limit.min(N_BATCH as usize) {
            return Err(BackendError::InvalidRequest);
        }
        total += tokens.len() as u64;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|_| BackendError::InvalidRequest)?;
        context.clear_kv_cache();
        context
            .decode(&mut batch)
            .map_err(|_| BackendError::Crashed)?;
        let raw = match context.embeddings_seq_ith(0) {
            Ok(values) => values.to_vec(),
            // Models without pooling expose one vector per token; the last one represents the text.
            Err(_) => context
                .embeddings_ith(batch.n_tokens() - 1)
                .map_err(|_| BackendError::Crashed)?
                .to_vec(),
        };
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        vectors.push(if norm > 0.0 {
            raw.iter().map(|v| v / norm).collect()
        } else {
            raw
        });
    }
    Ok(Embeddings {
        vectors,
        input_tokens: total,
    })
}

// ---------------------------------------------------------------------------------------------
// Chat generation
// ---------------------------------------------------------------------------------------------

struct Sequence {
    tx: mpsc::Sender<Result<Chunk, BackendError>>,
    sampler: LlamaSampler,
    parser: OutputParser,
    format: Format,
    stops: Vec<String>,
    /// Text held back because it may be the start of a stop sequence.
    held: String,
    /// Bytes of an incomplete UTF-8 character.
    pending_bytes: Vec<u8>,
    prompt: Vec<LlamaToken>,
    /// Tokens of the prompt (for the context limit and the usage); for a prompt with images the
    /// `prompt` list stays empty because the projector library decoded it.
    prompt_len: usize,
    /// Position of the first generated token (differs from `prompt_len` for M-RoPE models).
    base_pos: usize,
    /// The prompt held images: the slot's cache is dropped when the answer is done.
    vision: bool,
    /// Prompt tokens already decoded into this slot's cache.
    fed: usize,
    /// Sampled token that still has to be decoded to get the next one.
    next: Option<LlamaToken>,
    generated: usize,
    max_new: usize,
    /// Positions available to this sequence.
    limit: usize,
    finish: Option<&'static str>,
    /// A stop sequence ended the answer; whatever the parser still holds is dropped.
    stop_matched: bool,
    /// The client went away or the stream failed: no final chunk is sent.
    abandoned: bool,
}

#[derive(Default)]
struct Slot {
    /// Tokens whose keys and values are in the cache for this slot (for prefix reuse).
    cached: Vec<LlamaToken>,
    seq: Option<Sequence>,
}

struct Prompting {
    template: ChatTemplate,
    format: Format,
}

fn token_text(model: &LlamaModel, token: LlamaToken) -> String {
    model
        .vocab()
        .text(token)
        .and_then(|text| text.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

fn prompting(model: &LlamaModel) -> Option<Prompting> {
    let source = model
        .chat_template(None)
        .ok()
        .and_then(|t| t.to_str().ok().map(str::to_owned))
        .unwrap_or_else(|| FALLBACK_TEMPLATE.to_owned());
    let vocab = model.vocab();
    let template = ChatTemplate::new(
        &source,
        &token_text(model, vocab.bos()),
        &token_text(model, vocab.eos()),
    )
    .map_err(|error| tracing::error!(%error, "The chat template of the model cannot be used"))
    .ok()?;
    Some(Prompting {
        template,
        format: detect_format(&source),
    })
}

fn chat_loop(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    mut vision: Option<&mut MtmdContext>,
    setup: &Setup,
    commands: &std_mpsc::Receiver<Cmd>,
) {
    let Some(prompting) = prompting(model) else {
        return;
    };
    let parallel = setup.parallel.max(1) as usize;
    let limit = (setup.context as usize / parallel).max(1);
    let mut slots: Vec<Slot> = (0..parallel).map(|_| Slot::default()).collect();
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
    let mut stopping = false;

    loop {
        // 1. Intake: block while idle, otherwise only look.
        let idle = pending.is_empty() && slots.iter().all(|s| s.seq.is_none());
        let mut next = if idle {
            commands.recv().ok()
        } else {
            commands.try_recv().ok()
        };
        if idle && next.is_none() {
            break;
        }
        while let Some(cmd) = next {
            match cmd {
                Cmd::Stop => stopping = true,
                Cmd::Generate(job) => pending.push_back(job),
                Cmd::Embed(job) => {
                    let _ = job.reply.send(Err(BackendError::InvalidRequest));
                }
            }
            next = commands.try_recv().ok();
        }
        if stopping {
            break;
        }

        // 2. Admission of waiting requests into free slots.
        while !pending.is_empty() && slots.iter().any(|s| s.seq.is_none()) {
            let Some(job) = pending.pop_front() else {
                break;
            };
            admit(
                model,
                context,
                vision.as_deref_mut(),
                &prompting,
                limit,
                &mut slots,
                job,
            );
        }

        // 3. One batch: prompt chunks and one generation step per sequence.
        batch.clear();
        let mut budget = N_BATCH as usize;
        let mut sample_at: Vec<(usize, i32)> = Vec::new();
        let mut blocked = false;
        for (index, slot) in slots.iter_mut().enumerate() {
            let Some(seq) = slot.seq.as_mut() else {
                continue;
            };
            if seq.tx.is_closed() {
                seq.abandoned = true;
                seq.finish = Some("stop");
                continue;
            }
            if seq.finish.is_some() {
                continue;
            }
            if seq.tx.capacity() < MIN_ROOM {
                blocked = true;
                continue;
            }
            if seq.fed < seq.prompt.len() {
                let take = (seq.prompt.len() - seq.fed).min(budget);
                for offset in 0..take {
                    let at = seq.fed + offset;
                    let last = at + 1 == seq.prompt.len();
                    if batch
                        .add(seq.prompt[at], at as i32, &[index as i32], last)
                        .is_err()
                    {
                        seq.finish = Some("length");
                        break;
                    }
                    slot.cached.push(seq.prompt[at]);
                    if last {
                        sample_at.push((index, batch.n_tokens() - 1));
                    }
                }
                seq.fed += take;
                budget -= take;
            } else if let Some(token) = seq.next.take() {
                if budget == 0 {
                    seq.next = Some(token);
                    continue;
                }
                let at = seq.base_pos + seq.generated - 1;
                if batch.add(token, at as i32, &[index as i32], true).is_err() {
                    seq.finish = Some("length");
                    continue;
                }
                slot.cached.push(token);
                sample_at.push((index, batch.n_tokens() - 1));
                budget -= 1;
            }
        }

        // 4. Decode and sample.
        if batch.n_tokens() > 0 {
            if let Err(error) = context.decode(&mut batch) {
                tracing::error!(%error, "Decoding failed");
                for slot in &mut slots {
                    if let Some(seq) = slot.seq.take() {
                        let _ = seq.tx.try_send(Err(BackendError::Crashed));
                    }
                    slot.cached.clear();
                }
                context.clear_kv_cache();
                continue;
            }
            for (index, at) in sample_at {
                let Some(seq) = slots[index].seq.as_mut() else {
                    continue;
                };
                let token = seq.sampler.sample(context, at);
                accept(model, seq, token);
            }
        } else if blocked {
            // Every runnable sequence waits for its client to read; do not spin.
            std::thread::sleep(Duration::from_millis(2));
        }

        // 5. Finish sequences.
        for (index, slot) in slots.iter_mut().enumerate() {
            if slot.seq.as_ref().is_some_and(|s| s.finish.is_some())
                && let Some(seq) = slot.seq.take()
            {
                // The cache of a prompt with images is no prefix for anything else.
                if seq.vision {
                    slot.cached.clear();
                    let _ = context.clear_kv_cache_seq(Some(index as u32), None, None);
                }
                finish(seq);
            }
        }
    }
    // Requests that never started or are still running end with an error instead of silence.
    for job in pending {
        let _ = job.admitted.send(Err(BackendError::Crashed));
    }
    for slot in &mut slots {
        if let Some(seq) = slot.seq.take() {
            let _ = seq.tx.try_send(Err(BackendError::Cancelled));
        }
    }
}

fn number(body: &Value, key: &str) -> Option<f64> {
    body.get(key).and_then(Value::as_f64)
}

/// Builds the sampler chain from the OpenAI request fields.
fn sampler(model: &LlamaModel, body: &Value) -> Result<LlamaSampler, BackendError> {
    let temperature = number(body, "temperature").unwrap_or(0.8).max(0.0) as f32;
    let top_p = number(body, "top_p").unwrap_or(1.0).clamp(0.0, 1.0) as f32;
    let top_k = number(body, "top_k").map_or(40, |k| k as i32);
    let min_p = number(body, "min_p").unwrap_or(0.05).clamp(0.0, 1.0) as f32;
    let frequency = number(body, "frequency_penalty").unwrap_or(0.0) as f32;
    let presence = number(body, "presence_penalty").unwrap_or(0.0) as f32;
    let seed = body.get("seed").and_then(Value::as_u64).map_or_else(
        || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        },
        |s| s as u32,
    );
    let mut chain = Vec::new();
    if let Some(schema) = body["response_format"]
        .get("json_schema")
        .and_then(|j| j.get("schema"))
        .filter(|_| body["response_format"]["type"] == "json_schema")
    {
        let grammar = llama_cpp_2::json_schema_to_grammar(&schema.to_string())
            .map_err(|_| BackendError::InvalidRequest)?;
        chain.push(
            LlamaSampler::grammar(model, &grammar, "root")
                .map_err(|_| BackendError::InvalidRequest)?,
        );
    }
    if frequency != 0.0 || presence != 0.0 {
        chain.push(LlamaSampler::penalties(
            model.n_vocab(),
            64,
            1.0,
            frequency,
            presence,
        ));
    }
    if temperature <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        if top_k > 0 {
            chain.push(LlamaSampler::top_k(top_k));
        }
        if top_p < 1.0 {
            chain.push(LlamaSampler::top_p(top_p, 1));
        }
        if min_p > 0.0 {
            chain.push(LlamaSampler::min_p(min_p, 1));
        }
        chain.push(LlamaSampler::temp(temperature));
        chain.push(LlamaSampler::dist(seed));
    }
    Ok(LlamaSampler::chain_simple(chain))
}

fn stop_sequences(body: &Value) -> Vec<String> {
    match body.get("stop") {
        Some(Value::String(one)) if !one.is_empty() => vec![one.clone()],
        Some(Value::Array(many)) => many
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .take(8)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn admit(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    vision: Option<&mut MtmdContext>,
    prompting: &Prompting,
    limit: usize,
    slots: &mut [Slot],
    job: Job,
) {
    let refuse = |job: Job, reason: &str| {
        tracing::warn!(reason, "A request was refused");
        let _ = job.admitted.send(Err(BackendError::InvalidRequest));
    };
    let Some(messages) = job.body["messages"].as_array() else {
        return refuse(job, "messages are missing");
    };
    let tools = job
        .body
        .get("tools")
        .filter(|_| job.body["tool_choice"] != "none");
    let effort = job.reasoning.as_deref();
    // Images: the messages are flattened to text with one marker per image.
    let extracted = if has_images(messages) {
        match extract(messages, mtmd_default_marker()) {
            Ok(extracted) => Some(extracted),
            Err(error) => {
                tracing::warn!(%error, "The images of a request were refused");
                return refuse(job, "images refused");
            }
        }
    } else {
        None
    };
    let messages: &[Value] = extracted.as_ref().map_or(messages, |e| &e.messages);
    let options = RenderOptions {
        messages,
        tools,
        add_generation_prompt: true,
        enable_thinking: effort.map(|e| e != "none"),
        reasoning_effort: effort.filter(|e| *e != "none"),
    };
    let text = match prompting.template.render(&options) {
        Ok(text) => text,
        Err(error) => {
            tracing::warn!(%error, "The chat template rejected the request");
            return refuse(job, "template error");
        }
    };
    if let Some(extracted) = &extracted {
        let Some(mtmd) = vision else {
            return refuse(job, "the model cannot read images");
        };
        return admit_with_images(
            model,
            context,
            mtmd,
            limit,
            slots,
            job,
            &text,
            &extracted.images,
            prompting,
        );
    }
    // The template already contains the special tokens (BOS and so on) as text.
    let prompt = model.vocab().tokenize(text.as_bytes(), false, true);
    if prompt.is_empty() || prompt.len() >= limit {
        return refuse(job, "prompt does not fit the context window");
    }
    let sampler = match sampler(model, &job.body) {
        Ok(sampler) => sampler,
        Err(_) => return refuse(job, "sampling parameters rejected"),
    };
    let requested = job.body["max_completion_tokens"]
        .as_u64()
        .or_else(|| job.body["max_tokens"].as_u64())
        .map_or(usize::MAX, |n| n as usize);
    let max_new = requested.min(limit - prompt.len()).max(1);

    // Prefer the idle slot whose cache shares the longest prefix with this prompt.
    let common = |slot: &Slot| {
        slot.cached
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count()
    };
    let Some(index) = (0..slots.len())
        .filter(|i| slots[*i].seq.is_none())
        .max_by_key(|i| common(&slots[*i]))
    else {
        return refuse(job, "no free slot");
    };
    // At least the last prompt token must be decoded to obtain logits.
    let mut keep = common(&slots[index]).min(prompt.len() - 1);
    let removed = context.kv_cache_seq_rm(index as i32, Some(keep as u32), None);
    if removed.is_err() {
        // Recurrent and sliding-window models cannot drop a tail; start the slot from scratch.
        keep = 0;
        let _ = context.clear_kv_cache_seq(Some(index as u32), None, None);
    }
    slots[index].cached.truncate(keep);

    let prefilled = matches!(prompting.format, Format::Think | Format::QwenXml)
        && text.trim_end().ends_with("<think>");
    let sequence = Sequence {
        tx: job.tx,
        sampler,
        parser: OutputParser::new(prompting.format, prefilled),
        format: prompting.format,
        stops: stop_sequences(&job.body),
        held: String::new(),
        pending_bytes: Vec::new(),
        prompt_len: prompt.len(),
        base_pos: prompt.len(),
        vision: false,
        prompt,
        fed: keep,
        next: None,
        generated: 0,
        max_new,
        limit,
        finish: None,
        stop_matched: false,
        abandoned: false,
    };
    slots[index].seq = Some(sequence);
    let _ = job.admitted.send(Ok(()));
}

/// Admits a request whose prompt holds images: the projector library encodes the images and decodes
/// the whole prompt into the slot (it blocks the worker for that time), then the first token is
/// sampled from its last logits and generation goes on in the shared batches.
#[allow(clippy::too_many_arguments)]
fn admit_with_images(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    mtmd: &mut MtmdContext,
    limit: usize,
    slots: &mut [Slot],
    job: Job,
    text: &str,
    images: &[Vec<u8>],
    prompting: &Prompting,
) {
    let refuse = |job: Job, reason: &str| {
        tracing::warn!(reason, "A request with images was refused");
        let _ = job.admitted.send(Err(BackendError::InvalidRequest));
    };
    let mut bitmaps = Vec::with_capacity(images.len());
    for bytes in images {
        match MtmdBitmap::from_buffer(mtmd, bytes, false) {
            Ok(bitmap) => bitmaps.push(bitmap),
            Err(_) => return refuse(job, "an image cannot be decoded"),
        }
    }
    let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    // The template already contains the special tokens as text.
    let input = MtmdInputText {
        text: text.to_owned(),
        add_special: false,
        parse_special: true,
    };
    let chunks = match mtmd.tokenize(input, &refs) {
        Ok(chunks) => chunks,
        Err(error) => {
            tracing::warn!(%error, "The prompt with images cannot be tokenized");
            return refuse(job, "the images do not match the prompt");
        }
    };
    let total = chunks.total_tokens();
    if total == 0 || total >= limit {
        return refuse(job, "prompt does not fit the context window");
    }
    let sampler = match sampler(model, &job.body) {
        Ok(sampler) => sampler,
        Err(_) => return refuse(job, "sampling parameters rejected"),
    };
    let Some(index) = slots.iter().position(|s| s.seq.is_none()) else {
        return refuse(job, "no free slot");
    };
    // The cache of the slot cannot be reused: start from an empty slot.
    slots[index].cached.clear();
    let _ = context.clear_kv_cache_seq(Some(index as u32), None, None);
    let n_past = match chunks.eval_chunks(mtmd, context, 0, index as i32, N_BATCH as i32, true) {
        Ok(n_past) => n_past,
        Err(error) => {
            tracing::error!(%error, "The images could not be decoded into the context");
            let _ = context.clear_kv_cache_seq(Some(index as u32), None, None);
            return refuse(job, "the images could not be processed");
        }
    };
    let requested = job.body["max_completion_tokens"]
        .as_u64()
        .or_else(|| job.body["max_tokens"].as_u64())
        .map_or(usize::MAX, |n| n as usize);
    let max_new = requested.min(limit - total).max(1);
    let prefilled = matches!(prompting.format, Format::Think | Format::QwenXml)
        && text.trim_end().ends_with("<think>");
    let mut sequence = Sequence {
        tx: job.tx,
        sampler,
        parser: OutputParser::new(prompting.format, prefilled),
        format: prompting.format,
        stops: stop_sequences(&job.body),
        held: String::new(),
        pending_bytes: Vec::new(),
        prompt: Vec::new(),
        prompt_len: total,
        base_pos: usize::try_from(n_past).unwrap_or(total),
        vision: true,
        fed: 0,
        next: None,
        generated: 0,
        max_new,
        limit,
        finish: None,
        stop_matched: false,
        abandoned: false,
    };
    // The last logits of the decoded prompt give the first token.
    let token = sequence.sampler.sample(context, -1);
    accept(model, &mut sequence, token);
    slots[index].seq = Some(sequence);
    let _ = job.admitted.send(Ok(()));
}

/// Handles a freshly sampled token: end of generation, text output, limits.
fn accept(model: &LlamaModel, seq: &mut Sequence, token: LlamaToken) {
    let vocab = model.vocab();
    if vocab.is_eog(token) {
        // Harmony ends a tool call with `<|call|>`, which llama.cpp also reports as an
        // end-of-generation token; the parser needs to see it to complete the call.
        if seq.format == Format::Harmony {
            let piece = vocab.token_to_piece(token, true, None);
            if piece == b"<|call|>" {
                push_text(seq, &piece);
            }
        }
        seq.finish = Some("stop");
        return;
    }
    seq.generated += 1;
    let piece = vocab.token_to_piece(token, true, None);
    if !push_text(seq, &piece) {
        seq.stop_matched = true;
        seq.finish = Some("stop");
        return;
    }
    if seq.generated >= seq.max_new || seq.prompt_len + seq.generated >= seq.limit {
        seq.finish = Some("length");
        return;
    }
    seq.next = Some(token);
}

/// Sends a chunk; a full channel is rare (the room check runs first) and then simply waits.
fn send(seq: &mut Sequence, chunk: Chunk) {
    match seq.tx.try_send(Ok(chunk)) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(item)) => {
            if seq.tx.blocking_send(item).is_err() {
                seq.abandoned = true;
            }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => seq.abandoned = true,
    }
}

/// Sends visible text unless a stop sequence ends it. Stop sequences apply to the visible
/// answer only (as in the OpenAI API): thinking and tool calls pass through untouched. A tail
/// that could still grow into a stop sequence is held back. Returns false once one matched.
fn visible(seq: &mut Sequence, text: &str) -> bool {
    let mut text = format!("{}{text}", std::mem::take(&mut seq.held));
    if seq.stops.is_empty() {
        if !text.is_empty() {
            send(seq, Chunk::Text(text));
        }
        return true;
    }
    let hit = seq
        .stops
        .iter()
        .filter_map(|stop| text.find(stop.as_str()))
        .min();
    if let Some(at) = hit {
        text.truncate(at);
        if !text.is_empty() {
            send(seq, Chunk::Text(text));
        }
        return false;
    }
    let mut keep = 0;
    for stop in &seq.stops {
        for (start, _) in text.char_indices() {
            let tail = &text[start..];
            if tail.len() < stop.len() && stop.starts_with(tail) {
                keep = keep.max(tail.len());
                break;
            }
        }
    }
    seq.held = text.split_off(text.len() - keep);
    if !text.is_empty() {
        send(seq, Chunk::Text(text));
    }
    true
}

/// Text held back for a stop sequence that never completed is real output.
fn flush_held(seq: &mut Sequence) {
    let held = std::mem::take(&mut seq.held);
    if !held.is_empty() {
        send(seq, Chunk::Text(held));
    }
}

/// Routes parsed chunks to the client. Returns false once a stop sequence matched.
fn dispatch(seq: &mut Sequence, chunks: Vec<Chunk>) -> bool {
    for chunk in chunks {
        match chunk {
            Chunk::Text(text) => {
                if !visible(seq, &text) {
                    return false;
                }
            }
            other => {
                flush_held(seq);
                send(seq, other);
            }
        }
    }
    true
}

/// Turns bytes into text (holding back an incomplete character), parses it into text, thinking
/// and tool calls and sends the result. Returns false once a stop sequence matched.
fn push_text(seq: &mut Sequence, bytes: &[u8]) -> bool {
    seq.pending_bytes.extend_from_slice(bytes);
    let decoded = match std::str::from_utf8(&seq.pending_bytes) {
        Ok(text) => {
            let text = text.to_owned();
            seq.pending_bytes.clear();
            text
        }
        Err(error) if error.error_len().is_none() => {
            let valid = error.valid_up_to();
            let text = String::from_utf8_lossy(&seq.pending_bytes[..valid]).into_owned();
            seq.pending_bytes.drain(..valid);
            text
        }
        Err(_) => {
            let text = String::from_utf8_lossy(&seq.pending_bytes).into_owned();
            seq.pending_bytes.clear();
            text
        }
    };
    if decoded.is_empty() {
        return true;
    }
    let chunks = seq.parser.push(&decoded);
    dispatch(seq, chunks)
}

/// Flushes the parser and held text, then sends the final chunk.
fn finish(mut seq: Sequence) {
    if seq.abandoned {
        return;
    }
    // After a stop-sequence match the rest of the answer is dropped on purpose.
    let chunks = seq.parser.finish();
    if !seq.stop_matched {
        dispatch(&mut seq, chunks);
    }
    flush_held(&mut seq);
    let reason = match seq.finish {
        Some("stop") if seq.parser.saw_tool_call() => "tool_calls",
        Some(reason) => reason,
        None => "stop",
    };
    let (input_tokens, output_tokens) = (seq.prompt_len as u64, seq.generated as u64);
    send(
        &mut seq,
        Chunk::Done {
            finish_reason: reason.to_owned(),
            input_tokens,
            output_tokens,
        },
    );
}
