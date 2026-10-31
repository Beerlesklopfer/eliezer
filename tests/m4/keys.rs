//! The API key file and the targeted configuration edits behind `--new-key` and `--set-hf-token`.
use eliezer_core::{
    config::Config,
    edit::{EditError, set_string},
    keyfile::{self, Entry, KeyError},
};

#[test]
fn generated_keys_are_openai_style_random_and_distinct() {
    let a = keyfile::generate().unwrap();
    let b = keyfile::generate().unwrap();
    assert_ne!(a, b);
    assert!(a.starts_with("sk-") && a.len() == 3 + 64, "{a}");
    assert!(a[3..].bytes().all(|c| c.is_ascii_hexdigit()));
    // 256 bits from the system source: not all digits equal, not a counter
    assert!(
        a[3..]
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 6
    );
}

#[test]
fn names_are_checked() {
    for good in ["multigpt", "a", "key-1", "My_key.2", &"x".repeat(40)] {
        assert!(keyfile::valid_name(good), "{good}");
    }
    for bad in ["", "a b", "a/b", "a\nb", "ä", "x;y", &"x".repeat(41)] {
        assert!(!keyfile::valid_name(bad), "{bad:?}");
    }
}

#[test]
fn keys_are_added_listed_and_revoked_with_their_names() {
    let text = keyfile::add("", "multigpt", "sk-aaaa").unwrap();
    assert!(
        text.starts_with("# Eliezer API keys"),
        "a new file gets a header: {text}"
    );
    let text = keyfile::add(&text, "claude", "sk-bbbb").unwrap();
    assert_eq!(
        keyfile::parse(&text),
        vec![
            Entry {
                name: Some("multigpt".into()),
                key: "sk-aaaa".into()
            },
            Entry {
                name: Some("claude".into()),
                key: "sk-bbbb".into()
            },
        ]
    );
    assert_eq!(
        keyfile::add(&text, "claude", "sk-cccc"),
        Err(KeyError::NameTaken)
    );
    assert_eq!(
        keyfile::add(&text, "bad name", "sk-cccc"),
        Err(KeyError::InvalidName)
    );
    let after = keyfile::remove(&text, "multigpt").unwrap();
    assert_eq!(keyfile::parse(&after).len(), 1);
    assert!(
        !after.contains("sk-aaaa") && !after.contains("multigpt"),
        "key and comment go: {after}"
    );
    assert!(
        after.contains("sk-bbbb") && after.contains("# Eliezer API keys"),
        "the rest stays: {after}"
    );
    assert_eq!(keyfile::remove(&after, "multigpt"), Err(KeyError::NotFound));
    // revoking the last key leaves a valid, empty file
    let empty = keyfile::remove(&after, "claude").unwrap();
    assert!(keyfile::parse(&empty).is_empty() && empty.starts_with("# Eliezer API keys"));
}

#[test]
fn keys_without_names_and_foreign_comments_survive() {
    // a hand-written file: bare keys, free comments, blank lines
    let text = "# my own notes\n\nsk-plain\n# name: named\nsk-named\n# another comment\nsk-after-comment\n";
    let entries = keyfile::parse(text);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].name, None);
    assert_eq!(entries[1].name.as_deref(), Some("named"));
    assert_eq!(
        entries[2].name, None,
        "a name does not leak to the next key"
    );
    let after = keyfile::remove(text, "named").unwrap();
    assert!(
        after.contains("# my own notes")
            && after.contains("sk-plain")
            && after.contains("sk-after-comment")
    );
    assert!(!after.contains("sk-named"));
    assert_eq!(keyfile::free_name(&entries), "key-1");
    let used = keyfile::add("", "key-1", "sk-x").unwrap();
    assert_eq!(keyfile::free_name(&keyfile::parse(&used)), "key-2");
}

#[test]
fn keys_are_masked_for_display() {
    let key = "sk-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let masked = keyfile::mask(key);
    assert_eq!(masked, "sk-…cdef");
    assert!(!masked.contains("0123"));
    assert_eq!(
        keyfile::mask("short"),
        "*****",
        "short strings are hidden completely"
    );
}

const BASE: &str = "[server]\nlisten = \"127.0.0.1:1234\"\n# api_keys_file = \"/old\"\n\n[gpu]\nvendor = \"auto\"\n\n[huggingface]\ncache_dir = \"/m/hf\"\n";

#[test]
fn a_string_is_set_in_the_right_section_and_everything_else_stays() {
    let out = set_string(BASE, "server", "api_keys_file", "/etc/eliezer/keys").unwrap();
    assert!(
        out.contains("[server]\napi_keys_file = \"/etc/eliezer/keys\"\nlisten"),
        "{out}"
    );
    assert!(out.contains("# api_keys_file = \"/old\""), "comments stay");
    assert!(out.contains("[gpu]\nvendor = \"auto\"") && out.contains("cache_dir = \"/m/hf\""));
    // the other section gets its own key without touching the first
    let out = set_string(&out, "huggingface", "token_file", "/etc/eliezer/hf-token").unwrap();
    assert!(
        out.contains("[huggingface]\ntoken_file = \"/etc/eliezer/hf-token\"\ncache_dir"),
        "{out}"
    );
    // setting again replaces the active line, it does not add a second one
    let again = set_string(&out, "server", "api_keys_file", "/other").unwrap();
    assert_eq!(
        again.matches("api_keys_file = ").count(),
        2,
        "one comment and one active line: {again}"
    );
    assert!(
        again.contains("api_keys_file = \"/other\"") && !again.contains("\"/etc/eliezer/keys\"")
    );
    // a missing section is appended
    let out = set_string("[gpu]\n", "server", "run_user", "eliezer").unwrap();
    assert_eq!(out, "[gpu]\n\n[server]\nrun_user = \"eliezer\"\n");
    // a key of the same name in another section is not touched
    let text = "[gpu]\napi_keys_file = \"gpu\"\n[server]\nlisten = \"127.0.0.1:1\"\n";
    let out = set_string(text, "server", "api_keys_file", "/k").unwrap();
    assert!(
        out.contains("[gpu]\napi_keys_file = \"gpu\"")
            && out.contains("[server]\napi_keys_file = \"/k\""),
        "{out}"
    );
}

#[test]
fn unsafe_values_are_refused_and_the_result_is_valid_configuration() {
    for bad in ["a\"b", "a\\b", "a\nb"] {
        assert_eq!(
            set_string(BASE, "server", "api_keys_file", bad),
            Err(EditError::UnsafeValue)
        );
    }
    let out = set_string(BASE, "server", "api_keys_file", "/etc/eliezer/keys").unwrap();
    let config = Config::parse(&out).unwrap();
    assert_eq!(
        config.server.api_keys_file.unwrap().to_str(),
        Some("/etc/eliezer/keys")
    );
}
