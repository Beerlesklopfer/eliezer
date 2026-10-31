use eliezer_backend::{supervisor::*, *};
use eliezer_core::scheduler::{Clock, Instance, State};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
#[derive(Default)]
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}
fn model(key: &str) -> Model {
    Model {
        instance: Instance {
            key: key.into(),
            state: State::Unloaded,
            bytes: 6,
            priority: 1,
            pinned: false,
            image: false,
            in_flight: 0,
            last_used_ms: 0,
        },
        default_spec: LoadSpec {
            model: key.into(),
            context: 1024,
            parallel: 1,
            bytes: 6,
            device: eliezer_core::config::DevicePolicy::Auto,
        },
        max_context: 4096,
    }
}
fn setup(config: FakeConfig) -> (Supervisor, Arc<TestClock>) {
    let clock = Arc::new(TestClock::default());
    (
        Supervisor::spawn(
            vec![model("a"), model("b")],
            Arc::new(FakeBackend::new(config)),
            clock.clone(),
            Settings {
                budget: 10,
                ram_budget: 0,
                idle_ms: 1000,
                wait_ms: 100,
                queue_max: 4,
                retry_ms: 30,
            },
        ),
        clock,
    )
}
async fn wait_state(s: &Supervisor, key: &str, state: State) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if s.snapshot()
                .await
                .unwrap()
                .iter()
                .any(|i| i.key == key && i.state == state)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn load_evict_idle_and_reload() {
    let (s, clock) = setup(FakeConfig::default());
    s.load(model("a").default_spec).await.unwrap();
    s.load(model("b").default_spec).await.unwrap();
    wait_state(&s, "a", State::Unloaded).await;
    clock.0.store(1001, Ordering::Relaxed);
    s.tick().await.unwrap();
    wait_state(&s, "b", State::Unloaded).await;
    s.load(model("a").default_spec).await.unwrap();
    let mut changed = model("a").default_spec;
    changed.context = 2048;
    assert_eq!(s.load(changed).await.unwrap().context, 2048);
    s.unload("a".into()).await.unwrap();
    wait_state(&s, "a", State::Unloaded).await;
    s.shutdown().await;
}
#[tokio::test]
async fn busy_wait_timeout_and_disconnect() {
    let (s, clock) = setup(FakeConfig::default());
    let lease = s.acquire(model("a").default_spec).await.unwrap();
    let task = tokio::spawn({
        let s = s.clone();
        async move { s.load(model("b").default_spec).await }
    });
    // A second command ensures that the task gets a chance to enqueue.
    for _ in 0..20 {
        tokio::task::yield_now().await;
        s.snapshot().await.unwrap();
    }
    assert!(!task.is_finished());
    clock.0.store(101, Ordering::Relaxed);
    s.tick().await.unwrap();
    assert_eq!(task.await.unwrap(), Err(Error::Timeout));
    let cancelled = tokio::spawn({
        let s = s.clone();
        async move { s.load(model("b").default_spec).await }
    });
    tokio::task::yield_now().await;
    cancelled.abort();
    let _ = cancelled.await;
    s.tick().await.unwrap();
    drop(lease);
    s.unload("a".into()).await.unwrap();
    s.load(model("b").default_spec).await.unwrap();
    s.shutdown().await;
}
#[tokio::test]
async fn failed_start_is_reported_and_retry_uses_clock() {
    let (s, clock) = setup(FakeConfig {
        fail_load: true,
        ..Default::default()
    });
    assert_eq!(s.load(model("a").default_spec).await, Err(Error::Backend));
    wait_state(&s, "a", State::Failed).await;
    clock.0.store(31, Ordering::Relaxed);
    s.tick().await.unwrap();
    wait_state(&s, "a", State::Unloaded).await;
    let mut unknown = model("a").default_spec;
    unknown.model = "unknown".into();
    assert_eq!(s.load(unknown).await, Err(Error::UnknownModel));
    s.shutdown().await;
}
fn placing(ram_budget: u64) -> Supervisor {
    let image = Model {
        instance: Instance {
            image: true,
            ..model("img").instance
        },
        default_spec: model("img").default_spec,
        max_context: 4096,
    };
    Supervisor::spawn(
        vec![model("a"), model("b"), image],
        Arc::new(FakeBackend::new(FakeConfig::default())),
        Arc::new(TestClock::default()),
        Settings {
            budget: 10,
            ram_budget,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    )
}
async fn place_of(s: &Supervisor, key: &str) -> eliezer_core::scheduler::Place {
    let status = s.status().await.unwrap();
    status.places.into_iter().find(|(k, _)| k == key).unwrap().1
}
#[tokio::test]
async fn a_language_model_goes_to_the_cpu_while_an_image_model_holds_the_gpu() {
    use eliezer_core::scheduler::Place;
    let s = placing(20);
    s.load(model("img").default_spec).await.unwrap();
    // 6 + 6 would not fit next to the image model; the language model takes the CPU instead.
    s.load(model("a").default_spec).await.unwrap();
    assert_eq!(place_of(&s, "a").await, Place::Cpu);
    assert_eq!(place_of(&s, "img").await, Place::Gpu);
    let status = s.status().await.unwrap();
    assert_eq!(status.ram_used, 6);
    // The image model stays loaded.
    wait_state(&s, "img", State::Loaded).await;
    s.shutdown().await;
}
#[tokio::test]
async fn without_an_image_model_the_gpu_is_used_and_the_cpu_model_stays_put() {
    use eliezer_core::scheduler::Place;
    let s = placing(20);
    s.load(model("a").default_spec).await.unwrap();
    assert_eq!(place_of(&s, "a").await, Place::Gpu);
    s.unload("a".into()).await.unwrap();
    wait_state(&s, "a", State::Unloaded).await;
    s.load(model("img").default_spec).await.unwrap();
    s.load(model("b").default_spec).await.unwrap();
    assert_eq!(place_of(&s, "b").await, Place::Cpu);
    // The image model goes away: b is not moved, it stays on the CPU until unloaded.
    s.unload("img".into()).await.unwrap();
    wait_state(&s, "img", State::Unloaded).await;
    assert_eq!(place_of(&s, "b").await, Place::Cpu);
    s.shutdown().await;
}
#[tokio::test]
async fn an_explicit_device_is_respected() {
    use eliezer_core::{config::DevicePolicy, scheduler::Place};
    let s = placing(20);
    let mut spec = model("a").default_spec;
    spec.device = DevicePolicy::Cpu;
    s.load(spec).await.unwrap();
    assert_eq!(place_of(&s, "a").await, Place::Cpu);
    s.shutdown().await;
}
#[tokio::test]
async fn without_a_ram_budget_nothing_moves_to_the_cpu() {
    use eliezer_core::scheduler::Place;
    let s = placing(0);
    s.load(model("img").default_spec).await.unwrap();
    // The old behavior: the language model needs the GPU; 6 + 6 > 10, so the image model is
    // evicted (same priority) or the load is rejected, but never placed on the CPU.
    let _ = s.load(model("a").default_spec).await;
    assert_ne!(place_of(&s, "a").await, Place::Cpu);
    s.shutdown().await;
}
#[tokio::test]
async fn too_little_ram_is_reported_for_the_cpu() {
    use eliezer_core::{config::DevicePolicy, scheduler::Place};
    let s = placing(4);
    let mut spec = model("a").default_spec;
    spec.device = DevicePolicy::Cpu;
    assert_eq!(
        s.load(spec).await,
        Err(Error::InsufficientMemory {
            needed: 6,
            free: 4,
            place: Place::Cpu
        })
    );
    s.shutdown().await;
}

/// A backend whose pictures fail; whether the program still answers is set by the test.
struct Failing {
    inner: FakeBackend,
    error: BackendError,
    alive: bool,
}
impl Backend for Failing {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        self.inner.start(spec)
    }
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        self.inner.stop(running)
    }
    fn health<'a>(&'a self, _: &'a Running) -> BackendFuture<'a, bool> {
        let alive = self.alive;
        Box::pin(async move { Ok(alive) })
    }
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, tokio::sync::mpsc::Receiver<Result<Chunk, BackendError>>> {
        self.inner.chat(running, request)
    }
    fn image<'a>(&'a self, _: &'a Running, _: ImageRequest) -> BackendFuture<'a, Vec<String>> {
        let error = self.error;
        Box::pin(async move { Err(error) })
    }
}
fn picture(key: &str) -> ImageRequest {
    ImageRequest {
        model: key.into(),
        prompt: "x".into(),
        count: 1,
        size: None,
        seed: None,
        references: Vec::new(),
    }
}
async fn failing_picture(error: BackendError, alive: bool) -> (Result<Vec<String>, Error>, State) {
    let backend = Arc::new(Failing {
        inner: FakeBackend::new(FakeConfig::default()),
        error,
        alive,
    });
    let s = Supervisor::spawn(
        vec![model("a")],
        backend.clone(),
        Arc::new(TestClock::default()),
        Settings {
            budget: 10,
            ram_budget: 0,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let lease = s.acquire(model("a").default_spec).await.unwrap();
    let result = s.image(&lease, picture("a"), backend.as_ref()).await;
    // let the supervisor handle a crash report, if there is one
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let state = s
        .snapshot()
        .await
        .unwrap()
        .iter()
        .find(|i| i.key == "a")
        .unwrap()
        .state;
    s.shutdown().await;
    (result, state)
}
#[tokio::test]
async fn a_failed_picture_does_not_take_the_model_down_while_its_program_answers() {
    // a request the program refused: that request's error only
    let (result, state) = failing_picture(BackendError::InvalidRequest, true).await;
    assert_eq!(result, Err(Error::InvalidRequest));
    assert_eq!(state, State::Loaded);
    // a picture that could not be drawn, the program still answers: the model stays loaded for the other jobs
    let (result, state) = failing_picture(BackendError::Crashed, true).await;
    assert_eq!(result, Err(Error::Backend));
    assert_eq!(state, State::Loaded);
    // the program is gone: now it is a crash
    let (result, state) = failing_picture(BackendError::Crashed, false).await;
    assert_eq!(result, Err(Error::Backend));
    assert_eq!(state, State::Failed);
}
