use eliezer_backend::*;
fn spec() -> LoadSpec {
    LoadSpec {
        model: "test".into(),
        context: 1024,
        parallel: 1,
        bytes: 1000,
    }
}
fn request() -> ChatRequest {
    ChatRequest {
        model: "test".into(),
        input: "Hallo".into(),
        reasoning: None,
    }
}
#[tokio::test]
async fn load_stream_stop_reload() {
    let backend = FakeBackend::new(FakeConfig::default());
    let running = backend.start(&spec()).await.unwrap();
    let mut stream = backend.chat(&running, request()).await.unwrap();
    assert_eq!(
        stream.recv().await.unwrap().unwrap(),
        Chunk::Text("Prüfantwort".into())
    );
    assert!(matches!(
        stream.recv().await.unwrap().unwrap(),
        Chunk::Done { .. }
    ));
    assert!(stream.recv().await.is_none());
    backend.stop(&running).await.unwrap();
    assert!(!backend.health(&running).await.unwrap());
    assert!(matches!(
        backend.chat(&running, request()).await,
        Err(BackendError::Crashed)
    ));
    assert_ne!(
        backend.start(&spec()).await.unwrap().generation,
        running.generation
    );
}
#[tokio::test]
async fn configured_failures() {
    let backend = FakeBackend::new(FakeConfig {
        fail_load: true,
        ..Default::default()
    });
    assert!(matches!(
        backend.start(&spec()).await,
        Err(BackendError::Unavailable)
    ));
    let backend = FakeBackend::new(FakeConfig {
        crash_after: Some(1),
        ..Default::default()
    });
    let running = backend.start(&spec()).await.unwrap();
    let mut stream = backend.chat(&running, request()).await.unwrap();
    assert!(stream.recv().await.unwrap().is_ok());
    assert_eq!(stream.recv().await.unwrap(), Err(BackendError::Crashed));
    assert!(!backend.health(&running).await.unwrap());
}
#[tokio::test]
async fn disconnect_cancels_producer() {
    let backend = FakeBackend::new(FakeConfig {
        chunk_delay: std::time::Duration::from_secs(60),
        ..Default::default()
    });
    let running = backend.start(&spec()).await.unwrap();
    let stream = backend.chat(&running, request()).await.unwrap();
    drop(stream);
    // Abbruch beendet nur die Anfrage, nicht die geladene Instanz.
    assert!(backend.health(&running).await.unwrap());
}
