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
