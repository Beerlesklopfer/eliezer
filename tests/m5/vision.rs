use eliezer_llamacpp::vision::*;
use serde_json::json;

const MARKER: &str = "<__media__>";
// 1x1 transparent PNG
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn image(url: &str) -> serde_json::Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}
fn data(b64: &str) -> String {
    format!("data:image/png;base64,{b64}")
}

#[test]
fn images_become_markers_and_their_bytes_are_kept_in_order() {
    let messages = vec![
        json!({"role": "system", "content": "Read the page."}),
        json!({"role": "user", "content": [
            {"type": "text", "text": "Page 1: "}, image(&data(PNG)),
            {"type": "text", "text": " and page 2: "}, image(&data(PNG)),
        ]}),
    ];
    assert!(has_images(&messages));
    let out = extract(&messages, MARKER).unwrap();
    assert_eq!(out.images.len(), 2);
    assert_eq!(&out.images[0][..4], b"\x89PNG");
    assert_eq!(out.messages[0]["content"], "Read the page.");
    assert_eq!(
        out.messages[1]["content"],
        "Page 1: <__media__> and page 2: <__media__>"
    );
    assert_eq!(out.messages[1]["role"], "user");
}

#[test]
fn requests_without_images_are_left_alone() {
    let messages = vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})];
    assert!(!has_images(&messages));
    assert!(!has_images(&[json!({"role": "user", "content": "plain"})]));
}

#[test]
fn anthropic_style_and_plain_string_image_parts_work() {
    let messages = vec![json!({"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        {"type": "image_url", "image_url": data(PNG)},
    ]})];
    let out = extract(&messages, MARKER).unwrap();
    assert_eq!(out.images.len(), 2);
    assert_eq!(out.messages[0]["content"], "<__media__><__media__>");
}

#[test]
fn remote_links_are_never_followed() {
    for url in [
        "http://127.0.0.1:1234/x.png",
        "https://example.com/a.png",
        "file:///etc/passwd",
        "ftp://x/y",
        "data:image/png,rawtext",
        "data:image/png;charset=utf-8,abc",
        "//example.com/a.png",
    ] {
        let messages = vec![json!({"role": "user", "content": [image(url)]})];
        assert_eq!(
            extract(&messages, MARKER).err(),
            Some(VisionError::NotInline),
            "{url}"
        );
    }
}

#[test]
fn bad_data_too_many_and_too_large_images_are_refused() {
    let one = |part| vec![json!({"role": "user", "content": [part]})];
    assert_eq!(
        extract(&one(image(&data("not base64 !!!"))), MARKER).err(),
        Some(VisionError::BadData)
    );
    let many: Vec<_> = (0..=MAX_IMAGES).map(|_| image(&data(PNG))).collect();
    assert_eq!(
        extract(&[json!({"role": "user", "content": many})], MARKER).err(),
        Some(VisionError::TooMany)
    );
    // 21 MB of base64 text decode to more than the limit of one image
    let huge = "A".repeat(MAX_IMAGE_BYTES / 3 * 4 + 1024);
    assert_eq!(
        extract(&one(image(&data(&huge))), MARKER).err(),
        Some(VisionError::TooLarge)
    );
    assert_eq!(
        extract(
            &one(json!({"type": "input_audio", "input_audio": {}})),
            MARKER
        )
        .err(),
        Some(VisionError::UnsupportedPart)
    );
}
