use eliezer_hub::*;
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

fn file(name: &str, bytes: u64) -> RepoFile {
    RepoFile {
        name: name.into(),
        bytes,
    }
}
fn detail(architecture: Option<&str>, files: Vec<RepoFile>) -> Detail {
    Detail {
        repo: "org/model-GGUF".into(),
        sha: "abc123".into(),
        task: Some("text-generation".into()),
        license: Some("apache-2.0".into()),
        gated: false,
        downloads: 10,
        architecture: architecture.map(str::to_owned),
        files,
    }
}

#[test]
fn quantizations_are_read_from_file_names() {
    for (name, quant) in [
        ("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", Some("Q4_K_M")),
        ("qwen-image-2.1-UC-Q4_K_M.gguf", Some("Q4_K_M")),
        ("model-IQ4_XS.gguf", Some("IQ4_XS")),
        ("gpt-oss-20b-MXFP4.gguf", Some("MXFP4")),
        ("mmproj-model-BF16.gguf", Some("BF16")),
        ("model-F16.gguf", Some("F16")),
        ("model-Q8_0.gguf", Some("Q8_0")),
        ("README.md", None),
        // a quantization name inside a longer word is not one
        ("modelQ4_K_Mx.gguf", None),
    ] {
        assert_eq!(quantization(name), quant, "{name}");
    }
}

#[test]
fn repositories_are_written_organisation_slash_name() {
    assert!(valid_repo("abenzerps/Qwen-Image-2.1-Uncensored-Quantized"));
    assert!(valid_repo("lmstudio-community/olmOCR-2-7B-1025-GGUF"));
    for bad in [
        "",
        "x",
        "a/",
        "/b",
        "a/b/c",
        "a/../b",
        "a/.hidden",
        "a b/c",
        "a/b?x=1",
        "a/b#c",
        "a\\b/c",
    ] {
        assert!(!valid_repo(bad), "{bad:?}");
    }
}

#[test]
fn variants_are_grouped_and_judged_against_the_budgets() {
    let d = detail(
        Some("llama"),
        vec![
            file("README.md", 10),
            file("m-Q4_K_M.gguf", 4_000_000_000),
            file("m-Q8_0.gguf", 8_000_000_000),
            file("m-F16-00001-of-00002.gguf", 10_000_000_000),
            file("m-F16-00002-of-00002.gguf", 10_000_000_000),
            file("mmproj-m-F16.gguf", 1_000_000_000),
            file("sub/other-Q4_K_M.gguf", 1),
        ],
    );
    let a = assess(
        &d,
        Budgets {
            gpu: 12_000_000_000,
            ram: 40_000_000_000,
        },
    );
    assert_eq!(a.support, Support::Verified);
    assert_eq!(a.backend, Some("llamacpp"));
    assert_eq!(a.projector.as_ref().map(|p| p.bytes), Some(1_000_000_000));
    let by = |name: &str| a.variants.iter().find(|v| v.name == name).unwrap();
    // file + 20 % + projector
    assert_eq!(
        by("m-Q4_K_M.gguf").estimate_bytes,
        4_800_000_000 + 1_000_000_000
    );
    assert_eq!(by("m-Q4_K_M.gguf").fit, Fit::Gpu);
    assert_eq!(
        by("m-Q8_0.gguf").fit,
        Fit::Gpu,
        "9.6 + 1.0 = 10.6 GB is within 12 GB"
    );
    // the two parts of the split file count together
    let split = by("m-F16.gguf");
    assert_eq!((split.parts, split.bytes), (2, 20_000_000_000));
    assert_eq!(split.fit, Fit::Cpu);
    assert_eq!(
        a.variants.len(),
        3,
        "the projector and files in folders are no variants"
    );
    let small = assess(&d, Budgets { gpu: 1, ram: 0 });
    assert!(small.variants.iter().all(|v| v.fit == Fit::TooLarge));
}

#[test]
fn architectures_map_to_runtimes_honestly() {
    let level = |a: Option<&str>| assess(&detail(a, vec![]), Budgets::default()).support;
    assert_eq!(level(Some("qwen3moe")), Support::Verified);
    assert_eq!(level(Some("gpt-oss")), Support::Verified);
    assert_eq!(level(Some("gemma3")), Support::Expected);
    assert_eq!(level(Some("mamba-weird")), Support::Unsupported);
    assert_eq!(level(None), Support::Unknown);
    let image = assess(&detail(Some("qwen_image21"), vec![]), Budgets::default());
    assert_eq!(
        (image.support, image.backend),
        (Support::Verified, Some("sdcpp"))
    );
}

#[test]
fn text_encoders_and_vae_are_required_files_of_image_models() {
    let d = detail(
        Some("qwen_image21"),
        vec![
            file("qwen-image-2.1-UC-Q4_K_M.gguf", 4_300_000_000),
            file(
                "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                9_000_000_000,
            ),
            file("vae/qwen_image_2.1_vae_bf16.safetensors", 700_000_000),
            file("assets/benchmark.png", 5),
        ],
    );
    let a = assess(&d, Budgets::default());
    assert_eq!(a.required.len(), 2);
    assert_eq!(a.variants.len(), 1);
    // 4.3 GB + 20 % + the text encoder 9.0 GB + the VAE 0.7 GB
    assert_eq!(
        a.variants[0].estimate_bytes,
        5_160_000_000 + 9_000_000_000 + 700_000_000
    );
}

#[tokio::test]
async fn the_fake_hub_filters_and_records_queries() {
    let hub = FakeHub {
        summaries: vec![Summary {
            repo: "a/Qwen-GGUF".into(),
            downloads: 1,
            likes: 1,
            task: Some("text-generation".into()),
            license: Some("mit".into()),
            gguf: true,
        }],
        ..FakeHub::default()
    };
    let query = Query {
        text: "qwen".into(),
        task: Task::Llm,
        format: Format::Gguf,
        license: Some("mit".into()),
        limit: 5,
    };
    assert_eq!(hub.search(&query).await.unwrap().len(), 1);
    let other = Query {
        license: Some("apache-2.0".into()),
        ..query
    };
    assert!(hub.search(&other).await.unwrap().is_empty());
    assert_eq!(hub.queries.lock().unwrap().len(), 2);
    assert_eq!(hub.detail("x/y").await, Err(HubError::NotFound));
}

/// A one-shot HTTP server that answers with `status` and `body` and reports the request line.
async fn canned(status: &str, body: &str) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buffer = vec![0_u8; 4096];
        let n = stream.read(&mut buffer).await.unwrap();
        stream.write_all(reply.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&buffer[..n])
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned()
    });
    (base, handle)
}

#[tokio::test]
async fn the_real_client_asks_huggingface_the_way_it_was_probed() {
    let (base, request) = canned(
        "200 OK",
        r#"[{"id":"lmstudio-community/olmOCR-2-7B-1025-GGUF","downloads":10228,"likes":5,
            "tags":["gguf","license:apache-2.0"],"pipeline_tag":"image-text-to-text"},
           {"id":"bad id/with space","downloads":1,"tags":[]}]"#,
    )
    .await;
    let client = HfClient::new(&base, Duration::from_secs(5)).unwrap();
    let hits = client
        .search(&Query {
            text: "olmocr".into(),
            task: Task::Vision,
            format: Format::Gguf,
            license: Some("apache-2.0".into()),
            limit: 3,
        })
        .await
        .unwrap();
    let line = request.await.unwrap();
    assert!(line.starts_with("GET /api/models?"), "{line}");
    for part in [
        "search=olmocr",
        "filter=gguf",
        "filter=license%3Aapache-2.0",
        "pipeline_tag=image-text-to-text",
        "sort=downloads",
        "direction=-1",
        "limit=3",
    ] {
        assert!(line.contains(part), "{part} missing in {line}");
    }
    assert_eq!(
        hits.len(),
        1,
        "a repository name that is not valid is dropped"
    );
    assert_eq!(hits[0].license.as_deref(), Some("apache-2.0"));
    assert_eq!(hits[0].task.as_deref(), Some("image-text-to-text"));
    assert!(hits[0].gguf);
}

#[tokio::test]
async fn the_real_client_reads_details_and_maps_errors() {
    let (base, request) = canned(
        "200 OK",
        r#"{"sha":"7959e658","pipeline_tag":"text-to-image","tags":["license:other"],"gated":false,
            "downloads":5,"gguf":{"architecture":"qwen_image21","total":1},
            "siblings":[{"rfilename":"a.gguf","size":14230272800},{"rfilename":"README.md"}]}"#,
    )
    .await;
    let client = HfClient::new(&base, Duration::from_secs(5)).unwrap();
    let d = client
        .detail("abenzerps/Qwen-Image-2.1-Uncensored-Quantized")
        .await
        .unwrap();
    let line = request.await.unwrap();
    assert!(
        line.starts_with(
            "GET /api/models/abenzerps/Qwen-Image-2.1-Uncensored-Quantized?blobs=true"
        ),
        "{line}"
    );
    assert_eq!(d.sha, "7959e658");
    assert_eq!(d.architecture.as_deref(), Some("qwen_image21"));
    assert_eq!(d.files[0].bytes, 14_230_272_800);
    assert_eq!(d.files[1].bytes, 0);
    assert!(!d.gated);

    for (status, error) in [
        ("404 Not Found", HubError::NotFound),
        ("401 Unauthorized", HubError::NotFound),
        ("429 Too Many Requests", HubError::Busy),
        ("500 Internal Server Error", HubError::BadResponse),
    ] {
        let (base, _) = canned(status, "{}").await;
        let client = HfClient::new(&base, Duration::from_secs(5)).unwrap();
        assert_eq!(client.detail("a/b").await, Err(error), "{status}");
    }
    let (base, _) = canned("200 OK", "not json").await;
    let client = HfClient::new(&base, Duration::from_secs(5)).unwrap();
    assert_eq!(client.detail("a/b").await, Err(HubError::BadResponse));
    // nothing listens: unreachable, not a panic
    let client = HfClient::new("http://127.0.0.1:1", Duration::from_secs(2)).unwrap();
    assert_eq!(client.detail("a/b").await, Err(HubError::Unreachable));
    // an invalid repository name never becomes a URL
    assert_eq!(client.detail("a/b/../c").await, Err(HubError::NotFound));
}
