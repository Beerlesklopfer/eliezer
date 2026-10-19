//! A single task manages instances; load and unload work runs outside of it.
use crate::{
    Backend, BackendError, ChatRequest, Chunk, EmbedRequest, Embeddings, ImageRequest, LoadSpec,
    Running,
};
use eliezer_core::{
    config::DevicePolicy,
    scheduler::{
        Clock, Instance, Place, Plan, Queue, Queued, State, choose_place, idle_candidates,
        image_priority, plan_load,
    },
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    UnknownModel,
    InvalidLoad,
    /// Requirement and the memory that would be free after permitted eviction, in bytes, in the
    /// pool (GPU or CPU memory) the model was meant for.
    InsufficientMemory {
        needed: u64,
        free: u64,
        place: Place,
    },
    QueueFull,
    Timeout,
    Busy,
    Backend,
    /// The request itself is unusable (for example longer than the context window);
    /// the model is fine.
    InvalidRequest,
    /// The model file is missing or unreadable.
    ModelFile,
    Stopped,
}
#[derive(Clone)]
pub struct Model {
    pub instance: Instance,
    pub default_spec: LoadSpec,
    pub max_context: u32,
}
#[derive(Clone)]
pub struct Settings {
    pub budget: u64,
    /// Memory for models placed on the CPU; 0 turns CPU placement off.
    pub ram_budget: u64,
    pub idle_ms: u64,
    pub wait_ms: u64,
    pub queue_max: usize,
    pub retry_ms: u64,
}
/// Everything the status endpoint needs, taken at one instant.
#[derive(Debug, Clone)]
pub struct StatusInfo {
    pub now_ms: u64,
    pub instances: Vec<Instance>,
    /// Where each model lives (the pool of its last or current load).
    pub places: Vec<(String, Place)>,
    /// Bytes the instances on the CPU hold.
    pub ram_used: u64,
    /// Waiting requests as (model, enqueued_ms), in queue order.
    pub queue: Vec<(String, u64)>,
}
#[derive(Clone)]
pub struct Supervisor {
    tx: mpsc::Sender<Command>,
}
pub struct Lease {
    pub running: Running,
    tx: Option<mpsc::Sender<Command>>,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let Some(tx) = self.tx.clone() else {
            return;
        };
        let key = self.running.spec.model.clone();
        // Releasing must keep working even when the command queue is full.
        tokio::spawn(async move {
            let _ = tx.send(Command::Release(key)).await;
        });
    }
}
struct Pending {
    spec: LoadSpec,
    active: bool,
    reply: oneshot::Sender<Result<Lease, Error>>,
}
enum Command {
    Load(Pending),
    Release(String),
    Unload(String, oneshot::Sender<Result<(), Error>>),
    Snapshot(oneshot::Sender<Vec<Instance>>),
    Loaded(oneshot::Sender<Vec<LoadSpec>>),
    Status(oneshot::Sender<StatusInfo>),
    Reconfigure(Vec<Model>, Settings, oneshot::Sender<()>),
    Tick(oneshot::Sender<()>),
    Started(String, Result<Running, BackendError>),
    Stopped(String, Result<(), BackendError>),
    Crash(String),
    Shutdown(oneshot::Sender<()>),
}
struct Entry {
    model: Model,
    /// The pool of the current or last load.
    place: Place,
    running: Option<Running>,
    failed_ms: u64,
    /// Removed from the configuration: refuses new requests and unloads once idle.
    retired: bool,
}
struct Actor {
    entries: BTreeMap<String, Entry>,
    backend: Arc<dyn Backend>,
    clock: Arc<dyn Clock>,
    settings: Settings,
    queue: Queue<Pending>,
    next_id: u64,
    tx: mpsc::Sender<Command>,
    unload: BTreeMap<String, Vec<oneshot::Sender<Result<(), Error>>>>,
}
impl Supervisor {
    pub fn spawn(
        models: Vec<Model>,
        backend: Arc<dyn Backend>,
        clock: Arc<dyn Clock>,
        settings: Settings,
    ) -> Self {
        let (tx, rx) = mpsc::channel(128);
        let actor = Actor {
            entries: models
                .into_iter()
                .map(|model| {
                    (
                        model.instance.key.clone(),
                        Entry {
                            model,
                            running: None,
                            place: Place::Gpu,
                            failed_ms: 0,
                            retired: false,
                        },
                    )
                })
                .collect(),
            backend,
            clock,
            queue: Queue::new(settings.queue_max),
            settings,
            next_id: 0,
            tx: tx.clone(),
            unload: BTreeMap::new(),
        };
        tokio::spawn(actor.run(rx));
        Self { tx }
    }
    async fn request(&self, spec: LoadSpec, active: bool) -> Result<Lease, Error> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Load(Pending {
                spec,
                active,
                reply,
            }))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)?
    }
    pub async fn load(&self, spec: LoadSpec) -> Result<LoadSpec, Error> {
        let lease = self.request(spec, false).await?;
        Ok(lease.running.spec.clone())
    }
    pub async fn acquire(&self, spec: LoadSpec) -> Result<Lease, Error> {
        self.request(spec, true).await
    }
    pub async fn unload(&self, key: String) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Unload(key, tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)?
    }
    pub async fn snapshot(&self) -> Result<Vec<Instance>, Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Snapshot(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    /// The configurations actually loaded (context length and parallelism).
    pub async fn loaded(&self) -> Result<Vec<LoadSpec>, Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Loaded(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    pub async fn status(&self) -> Result<StatusInfo, Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Status(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    /// Applies a reloaded configuration: new models appear, changed settings take effect,
    /// removed models refuse new requests and are unloaded as soon as they are idle.
    pub async fn reconfigure(&self, models: Vec<Model>, settings: Settings) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Reconfigure(models, settings, tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    pub async fn tick(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Tick(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    pub async fn chat(
        &self,
        lease: &Lease,
        request: ChatRequest,
        backend: &dyn Backend,
    ) -> Result<mpsc::Receiver<Result<Chunk, BackendError>>, Error> {
        match backend.chat(&lease.running, request).await {
            Ok(stream) => Ok(stream),
            Err(BackendError::InvalidRequest) => Err(Error::InvalidRequest),
            Err(_) => {
                let _ = self
                    .tx
                    .send(Command::Crash(lease.running.spec.model.clone()))
                    .await;
                Err(Error::Backend)
            }
        }
    }
    pub async fn embed(
        &self,
        lease: &Lease,
        request: EmbedRequest,
        backend: &dyn Backend,
    ) -> Result<Embeddings, Error> {
        match backend.embed(&lease.running, request).await {
            Ok(result) => Ok(result),
            Err(_) => {
                self.report_crash(lease.running.spec.model.clone()).await;
                Err(Error::Backend)
            }
        }
    }
    pub async fn image(
        &self,
        lease: &Lease,
        request: ImageRequest,
        backend: &dyn Backend,
    ) -> Result<Vec<String>, Error> {
        match backend.image(&lease.running, request).await {
            Ok(result) => Ok(result),
            Err(_) => {
                self.report_crash(lease.running.spec.model.clone()).await;
                Err(Error::Backend)
            }
        }
    }
    pub async fn report_crash(&self, key: String) {
        let _ = self.tx.send(Command::Crash(key)).await;
    }
    pub async fn shutdown(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Command::Shutdown(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }
}
impl Actor {
    fn snapshot(&self) -> Vec<Instance> {
        self.entries
            .values()
            .map(|e| e.model.instance.clone())
            .collect()
    }
    fn reconfigure(&mut self, models: Vec<Model>, settings: Settings) {
        let wanted: std::collections::BTreeSet<String> =
            models.iter().map(|m| m.instance.key.clone()).collect();
        for entry in self.entries.values_mut() {
            entry.retired = !wanted.contains(&entry.model.instance.key);
        }
        for model in models {
            match self.entries.get_mut(&model.instance.key) {
                Some(entry) => {
                    let idle =
                        matches!(entry.model.instance.state, State::Unloaded | State::Failed);
                    let instance = &mut entry.model.instance;
                    instance.priority = model.instance.priority;
                    instance.pinned = model.instance.pinned;
                    instance.image = model.instance.image;
                    if idle {
                        instance.bytes = model.instance.bytes;
                    }
                    entry.model.default_spec = model.default_spec;
                    entry.model.max_context = model.max_context;
                }
                None => {
                    self.entries.insert(
                        model.instance.key.clone(),
                        Entry {
                            model,
                            running: None,
                            place: Place::Gpu,
                            failed_ms: 0,
                            retired: false,
                        },
                    );
                }
            }
        }
        self.queue.set_max(settings.queue_max);
        self.settings = settings;
    }
    fn stop(&mut self, key: &str) {
        let Some(entry) = self.entries.get_mut(key) else {
            return;
        };
        if entry.model.instance.state != State::Loaded || entry.model.instance.in_flight > 0 {
            return;
        }
        let Some(running) = entry.running.take() else {
            return;
        };
        entry.model.instance.state = State::Unloading;
        let backend = self.backend.clone();
        let tx = self.tx.clone();
        let key = key.to_owned();
        tokio::spawn(async move {
            let result = backend.stop(&running).await;
            let _ = tx.send(Command::Stopped(key, result)).await;
        });
    }
    fn start(&mut self, mut spec: LoadSpec, place: Place) {
        let Some(entry) = self.entries.get_mut(&spec.model) else {
            return;
        };
        entry.place = place;
        // The backend is told where, never "auto".
        spec.device = place.device();
        entry.model.instance.state = State::Loading;
        entry.model.instance.bytes = spec.bytes;
        let backend = self.backend.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result = backend.start(&spec).await;
            let _ = tx.send(Command::Started(spec.model.clone(), result)).await;
        });
    }
    fn tick(&mut self) {
        let now = self.clock.now_ms();
        self.queue.remove_if(|item| item.value.reply.is_closed());
        for item in self
            .queue
            .remove_if(|item| now.saturating_sub(item.enqueued_ms) >= self.settings.wait_ms)
        {
            let _ = item.value.reply.send(Err(Error::Timeout));
        }
        for entry in self.entries.values_mut() {
            if entry.model.instance.state == State::Failed
                && entry.model.instance.in_flight == 0
                && now.saturating_sub(entry.failed_ms) >= self.settings.retry_ms
            {
                entry.model.instance.state = State::Unloaded;
            }
        }
        for key in idle_candidates(&self.snapshot(), now, self.settings.idle_ms) {
            self.stop(&key);
        }
        let retired: Vec<String> = self
            .entries
            .values()
            .filter(|e| e.retired)
            .map(|e| e.model.instance.key.clone())
            .collect();
        for key in retired {
            self.stop(&key);
            let gone = self.entries.get(&key).is_some_and(|e| {
                matches!(e.model.instance.state, State::Unloaded | State::Failed)
                    && e.model.instance.in_flight == 0
            });
            if gone {
                self.entries.remove(&key);
            }
        }
    }
    /// Plans a load inside one pool of memory: only the instances living there compete for its
    /// budget, and the exclusivity of diffusion models applies to the GPU only. Returns the plan and,
    /// for a rejection, the memory that would be free beside pinned or busy models.
    fn plan_in(&self, place: Place, request: &Instance) -> (Plan, u64) {
        let pool: Vec<Instance> = self
            .entries
            .values()
            .filter(|e| e.place == place)
            .map(|e| e.model.instance.clone())
            .collect();
        let budget = match place {
            Place::Gpu => self.settings.budget,
            Place::Cpu => self.settings.ram_budget,
        };
        let held =
            |i: &&Instance| matches!(i.state, State::Loaded | State::Loading | State::Unloading);
        let leases = pool
            .iter()
            .filter(held)
            .fold(0_u64, |sum, i| sum.saturating_add(i.bytes));
        let plan = plan_load(&pool, request, budget.saturating_sub(leases));
        // Free is what remains beside pinned or busy models.
        let fixed = pool
            .iter()
            .filter(held)
            .filter(|i| i.pinned || i.in_flight > 0 || i.state != State::Loaded)
            .fold(0_u64, |sum, i| sum.saturating_add(i.bytes));
        (plan, budget.saturating_sub(fixed))
    }
    fn drain(&mut self) {
        let preferred = self
            .entries
            .values()
            .find(|e| e.model.instance.state == State::Loaded && e.model.instance.in_flight > 0)
            .map(|e| e.model.instance.key.clone());
        let count = self.queue.len();
        for _ in 0..count {
            let Some((key, item)) = self.queue.pop(preferred.as_deref()) else {
                break;
            };
            if item.value.reply.is_closed() {
                continue;
            }
            let pending = item.value;
            let Some(entry) = self.entries.get(&key).filter(|e| !e.retired) else {
                let _ = pending.reply.send(Err(Error::UnknownModel));
                continue;
            };
            let state = entry.model.instance.state;
            let mut keep = true;
            let mut shortage = (pending.spec.bytes, 0, Place::Gpu);
            if self.unload.contains_key(&key) {
                // An explicit unload takes precedence over new requests.
            } else if state == State::Loaded {
                if let Some(running) = entry.running.as_ref() {
                    if running.spec.satisfies(&pending.spec) {
                        let running = running.clone();
                        if let Some(entry) = self.entries.get_mut(&key) {
                            if pending.active {
                                entry.model.instance.in_flight += 1;
                            }
                            entry.model.instance.last_used_ms = self.clock.now_ms();
                        }
                        let tx = pending.active.then(|| self.tx.clone());
                        let _ = pending.reply.send(Ok(Lease { running, tx }));
                        continue;
                    } else if entry.model.instance.in_flight == 0 {
                        self.stop(&key);
                    }
                }
            } else if state == State::Unloaded {
                let mut request = entry.model.instance.clone();
                request.bytes = pending.spec.bytes;
                let cpu_enabled = self.settings.ram_budget > 0;
                let image_waiting = self.queue.iter().any(|(model, _)| {
                    self.entries
                        .get(model)
                        .is_some_and(|e| e.model.instance.image)
                });
                let priority = image_priority(&self.snapshot(), image_waiting);
                let mut place =
                    choose_place(pending.spec.device, request.image, cpu_enabled, priority);
                let mut outcome = self.plan_in(place, &request);
                // A language model left to `auto` that cannot fit on the GPU, even after evicting
                // what may be evicted, uses the CPU instead.
                if matches!(outcome.0, Plan::Reject)
                    && place == Place::Gpu
                    && pending.spec.device == DevicePolicy::Auto
                    && cpu_enabled
                    && !request.image
                {
                    let on_cpu = self.plan_in(Place::Cpu, &request);
                    if !matches!(on_cpu.0, Plan::Reject) {
                        place = Place::Cpu;
                        outcome = on_cpu;
                    }
                }
                match outcome.0 {
                    Plan::Load => self.start(pending.spec.clone(), place),
                    Plan::LoadAfterUnload(keys) => {
                        for key in keys {
                            self.stop(&key);
                        }
                    }
                    Plan::Wait => {}
                    Plan::Reject => {
                        keep = false;
                        shortage = (request.bytes, outcome.1, place);
                    }
                }
            }
            if keep {
                let _ = self.queue.push(
                    key,
                    Queued {
                        value: pending,
                        id: item.id,
                        enqueued_ms: item.enqueued_ms,
                    },
                );
            } else {
                let (needed, free, place) = shortage;
                let _ = pending.reply.send(Err(Error::InsufficientMemory {
                    needed,
                    free,
                    place,
                }));
            }
        }
    }
    async fn run(mut self, mut rx: mpsc::Receiver<Command>) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            let command = tokio::select! { command = rx.recv() => command, _ = interval.tick() => { self.tick(); self.drain(); continue; } };
            let Some(command) = command else {
                break;
            };
            match command {
                Command::Load(pending) => {
                    let key = pending.spec.model.clone();
                    let error = match self.entries.get(&key).filter(|e| !e.retired) {
                        None => Some(Error::UnknownModel),
                        Some(e)
                            if pending.spec.context == 0
                                || pending.spec.context > e.model.max_context
                                || pending.spec.parallel == 0
                                || pending.spec.parallel > pending.spec.context =>
                        {
                            Some(Error::InvalidLoad)
                        }
                        _ => None,
                    };
                    if let Some(error) = error {
                        let _ = pending.reply.send(Err(error));
                    } else {
                        self.next_id = self.next_id.wrapping_add(1);
                        if let Err(item) = self.queue.push(
                            key,
                            Queued {
                                id: self.next_id,
                                enqueued_ms: self.clock.now_ms(),
                                value: pending,
                            },
                        ) {
                            let _ = item.value.reply.send(Err(Error::QueueFull));
                        }
                    }
                }
                Command::Release(key) => {
                    if let Some(e) = self.entries.get_mut(&key) {
                        e.model.instance.in_flight = e.model.instance.in_flight.saturating_sub(1);
                        e.model.instance.last_used_ms = self.clock.now_ms();
                    }
                    if self.unload.contains_key(&key) {
                        self.stop(&key);
                    }
                }
                Command::Unload(key, reply) => {
                    match self.entries.get(&key).map(|e| e.model.instance.state) {
                        None => {
                            let _ = reply.send(Err(Error::UnknownModel));
                        }
                        Some(State::Unloaded | State::Failed) => {
                            let _ = reply.send(Ok(()));
                        }
                        Some(_) => {
                            self.unload.entry(key.clone()).or_default().push(reply);
                            self.stop(&key);
                        }
                    }
                }
                Command::Started(key, result) => {
                    let failure = match &result {
                        Err(BackendError::ModelFile) => Error::ModelFile,
                        _ => Error::Backend,
                    };
                    if let Some(entry) = self.entries.get_mut(&key) {
                        match result {
                            Ok(running) => {
                                entry.running = Some(running);
                                entry.model.instance.state = State::Loaded;
                                entry.model.instance.last_used_ms = self.clock.now_ms();
                            }
                            Err(_) => {
                                entry.model.instance.state = State::Failed;
                                entry.failed_ms = self.clock.now_ms();
                            }
                        }
                    }
                    if self
                        .entries
                        .get(&key)
                        .is_some_and(|e| e.model.instance.state == State::Failed)
                    {
                        for item in self.queue.remove_if(|i| i.value.spec.model == key) {
                            let _ = item.value.reply.send(Err(failure));
                        }
                    }
                    if self.unload.contains_key(&key) {
                        self.stop(&key);
                    }
                }
                Command::Stopped(key, result) => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.model.instance.state = if result.is_ok() {
                            State::Unloaded
                        } else {
                            State::Failed
                        };
                        entry.failed_ms = self.clock.now_ms();
                    }
                    if let Some(replies) = self.unload.remove(&key) {
                        for reply in replies {
                            let _ = reply.send(result.map_err(|_| Error::Backend));
                        }
                    }
                }
                Command::Crash(key) => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.model.instance.state = State::Failed;
                        entry.running = None;
                        entry.failed_ms = self.clock.now_ms();
                    }
                }
                Command::Snapshot(reply) => {
                    let _ = reply.send(self.snapshot());
                }
                Command::Loaded(reply) => {
                    let specs = self
                        .entries
                        .values()
                        .filter(|e| e.model.instance.state == State::Loaded)
                        .filter_map(|e| e.running.as_ref().map(|r| r.spec.clone()))
                        .collect();
                    let _ = reply.send(specs);
                }
                Command::Status(reply) => {
                    let _ = reply.send(StatusInfo {
                        now_ms: self.clock.now_ms(),
                        instances: self.snapshot(),
                        places: self
                            .entries
                            .iter()
                            .map(|(key, e)| (key.clone(), e.place))
                            .collect(),
                        ram_used: self
                            .entries
                            .values()
                            .filter(|e| {
                                e.place == Place::Cpu
                                    && matches!(
                                        e.model.instance.state,
                                        State::Loaded | State::Loading | State::Unloading
                                    )
                            })
                            .fold(0_u64, |sum, e| sum.saturating_add(e.model.instance.bytes)),
                        queue: self
                            .queue
                            .iter()
                            .map(|(model, item)| (model.to_owned(), item.enqueued_ms))
                            .collect(),
                    });
                }
                Command::Reconfigure(models, settings, reply) => {
                    self.reconfigure(models, settings);
                    let _ = reply.send(());
                }
                Command::Tick(reply) => {
                    self.tick();
                    let _ = reply.send(());
                }
                Command::Shutdown(reply) => {
                    for item in self.queue.remove_if(|_| true) {
                        let _ = item.value.reply.send(Err(Error::Stopped));
                    }
                    for entry in self.entries.values_mut() {
                        if let Some(running) = entry.running.take() {
                            let _ = self.backend.stop(&running).await;
                        }
                    }
                    let _ = reply.send(());
                    break;
                }
            }
            self.drain();
        }
    }
}
