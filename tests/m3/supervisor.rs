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
    // Ein zweiter Befehl stellt sicher, dass der Task Gelegenheit zum Einreihen bekommt.
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
