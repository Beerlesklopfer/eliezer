use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{
    error::Error,
    fs,
    io::{self, Write},
    num::NonZeroU32,
    path::Path,
    time::{Duration, Instant},
};

type ProbeResult<T> = Result<T, Box<dyn Error>>;

// Signatur und NULL-Verhalten aus dem installierten hip_runtime_api.h (ROCm 7.2).
#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipFree(ptr: *mut std::ffi::c_void) -> i32;
}

fn memory() -> ProbeResult<u64> {
    Ok(
        fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_used")?
            .trim()
            .parse()?,
    )
}

fn generate(model: &LlamaModel, backend: &LlamaBackend, parallel: usize) -> ProbeResult<()> {
    let template = model.chat_template(None)?;
    let prompt = model.apply_chat_template(
        &template,
        &[LlamaChatMessage::new(
            "user".into(),
            "Antworte kurz: Was ist Rust?".into(),
        )?],
        true,
    )?;
    let vocab = model.vocab();
    let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
    let mut context = model.new_context(
        backend,
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(1024))
            .with_n_seq_max(u32::try_from(parallel)?),
    )?;
    let mut batch = LlamaBatch::new(1024, i32::try_from(parallel)?);
    for seq in 0..parallel {
        batch.add_sequence(&tokens, i32::try_from(seq)?, false)?;
    }
    let mut indices = (1..=parallel)
        .map(|seq| i32::try_from(seq * tokens.len() - 1))
        .collect::<Result<Vec<_>, _>>()?;
    let mut samplers = (0..parallel)
        .map(|_| LlamaSampler::greedy())
        .collect::<Vec<_>>();
    let mut counts = vec![0; parallel];
    let mut first = vec![None; parallel];
    let generation = Instant::now();
    context.decode(&mut batch)?;
    for step in 0..32 {
        batch.clear();
        for seq in 0..parallel {
            if indices[seq] < 0 {
                continue;
            }
            let token = samplers[seq].sample(&context, indices[seq]);
            if vocab.is_eog(token) {
                indices[seq] = -1;
                continue;
            }
            let elapsed = generation.elapsed().as_millis();
            first[seq].get_or_insert(elapsed);
            counts[seq] += 1;
            println!(
                "Teil Sequenz {seq}, {elapsed} ms: {:?}",
                String::from_utf8_lossy(&vocab.token_to_piece(token, false, None))
            );
            io::stdout().flush()?;
            indices[seq] = batch.n_tokens();
            batch.add(
                token,
                i32::try_from(tokens.len() + step)?,
                &[i32::try_from(seq)?],
                true,
            )?;
        }
        if batch.n_tokens() == 0 || step == 31 {
            break;
        }
        context.decode(&mut batch)?;
    }
    println!(
        "Parallelität {parallel}: Prompt je {} Token; Ausgabe {counts:?}; erstes Token {first:?} ms; Gesamtzeit {} ms; VRAM {} Byte",
        tokens.len(),
        generation.elapsed().as_millis(),
        memory()?
    );
    Ok(())
}

fn run(model_path: &Path) -> ProbeResult<()> {
    let initial = memory()?;
    // SICHERHEIT: hipFree(NULL) initialisiert laut HIP-Header nur die Laufzeit;
    // kein Speicher wird dereferenziert und kein fremder Zeiger freigegeben.
    let status = unsafe { hipFree(std::ptr::null_mut()) };
    if status != 0 {
        return Err(format!("HIP-Initialisierung fehlgeschlagen, Fehlercode {status}").into());
    }
    println!(
        "Nach HIP-Initialisierung: {} Byte; vor Initialisierung {initial} Byte",
        memory()?
    );
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    if !backend.supports_gpu_offload() {
        return Err("GPU-Auslagerung nicht verfügbar".into());
    }
    println!("GPU-Auslagerung verfügbar; VRAM vor Backend {initial} Byte");
    for round in 1..=3 {
        let before = memory()?;
        let start = Instant::now();
        let model = LlamaModel::load_from_file(
            &backend,
            model_path,
            &LlamaModelParams::default().with_n_gpu_layers(999),
        )?;
        println!(
            "Durchlauf {round}: Laden {} ms; VRAM vorher {before}, geladen {} Byte",
            start.elapsed().as_millis(),
            memory()?
        );
        generate(&model, &backend, if round == 3 { 2 } else { 1 })?;
        drop(model);
        let stop = Instant::now();
        while memory()? > before + 100_000_000 && stop.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(100));
        }
        let after = memory()?;
        println!(
            "Durchlauf {round}: VRAM nach Entladen {after} Byte nach {} ms; Restdifferenz {} Byte",
            stop.elapsed().as_millis(),
            i128::from(after) - i128::from(before)
        );
    }
    drop(backend);
    println!(
        "Nach Backend-Drop: {} Byte, Ausgangswert {initial} Byte",
        memory()?
    );
    Ok(())
}

fn main() -> ProbeResult<()> {
    let path = std::env::args().nth(1).ok_or("Modellpfad fehlt")?;
    run(Path::new(&path))
}

#[test]
#[ignore = "Benötigt eine AMD-GPU, ROCm und ein lokales GGUF-Modell in ELIEZER_TEST_MODEL"]
fn gpu_lifecycle() -> ProbeResult<()> {
    let path = std::env::var("ELIEZER_TEST_MODEL")?;
    run(Path::new(&path))
}
