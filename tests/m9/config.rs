//! Configuration of speech synthesis (`kind = "tts"`) models and of the voice settings of `[backend.audiocpp]`.
use eliezer_core::config::{Config, ModelKind, valid_voice_name};
use std::path::PathBuf;

const BASE: &str = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.audiocpp]\nprogram=\"/usr/bin/audiocpp_server\"\nport_range=[24700,24709]\n";

fn tts(extra: &str) -> String {
    format!(
        "{BASE}[[model]]\nkey=\"reader\"\nbackend=\"audiocpp\"\nkind=\"tts\"\nsource={{file=\"/var/lib/eliezer/models/audio-cpp/Chatterbox-GGUF/chatterbox-q8_0.gguf\"}}\nvram_gb=4.0\n{extra}"
    )
}
fn error_of(text: &str) -> String {
    Config::parse(text).unwrap_err().to_string()
}

#[test]
fn a_speech_synthesis_model_is_served_by_audiocpp_and_needs_a_file_and_a_memory_figure() {
    let config = Config::parse(&tts("")).unwrap();
    assert_eq!(config.model[0].kind, ModelKind::Tts);
    assert!(
        error_of(&tts("").replace("backend=\"audiocpp\"", "backend=\"sdcpp\"")).contains("backend")
    );
    // audiocpp serves music and speech synthesis only
    assert!(error_of(&tts("").replace("kind=\"tts\"", "kind=\"image\"")).contains("kind"));
    assert!(error_of(&tts("").replace("vram_gb=4.0\n", "")).contains("vram_gb"));
    assert!(
        error_of(&tts("").replace(
            "source={file=\"/var/lib/eliezer/models/audio-cpp/Chatterbox-GGUF/chatterbox-q8_0.gguf\"}",
            "source={hf=\"audio-cpp/audio.cpp-gguf\"}"
        ))
        .contains("source")
    );
    assert!(error_of(&tts("device=\"cpu\"\n")).contains("device"));
    // the backend must be configured
    assert!(
        error_of(&tts("").replace(
            "[backend.audiocpp]\nprogram=\"/usr/bin/audiocpp_server\"\nport_range=[24700,24709]\n",
            ""
        ))
        .contains("backend")
    );
}

#[test]
fn the_settings_of_a_speech_synthesis_model_are_checked() {
    let ok = tts(
        "extra={family=\"chatterbox\",language=\"de\",speed=1.1,max_passage_chars=300,pause_ms=250,paragraph_pause_ms=900,max_input_chars=50000,default_voice=\"mama\",tts_options={exaggeration=0.5,temperature=0.7,max_tokens=400}}\n",
    );
    let config = Config::parse(&ok).unwrap();
    let extra = config.model[0].extra.as_ref().unwrap();
    assert_eq!(extra.tts_options.as_ref().unwrap().len(), 3);
    for (setting, field) in [
        ("family=\"xtts\"", "family"),
        ("language=\"auto\"", "language"),
        ("language=\"Deutsch\"", "language"),
        ("speed=0.1", "speed"),
        ("speed=9.0", "speed"),
        ("max_passage_chars=10", "extra"),
        ("max_passage_chars=99999", "extra"),
        ("pause_ms=99999", "extra"),
        ("max_input_chars=0", "extra"),
        ("default_voice=\"../x\"", "default_voice"),
        ("default_voice=\"a b\"", "default_voice"),
        ("tts_options={unknown=1.0}", "tts_options"),
        // an option of the other family
        ("tts_options={top_k=5.0}", "tts_options"),
        ("tts_options={temperature=\"hot\"}", "extra"),
        ("duration_sec=30", "extra"),
        ("threads=4", "extra"),
    ] {
        let message = error_of(&tts(&format!("extra={{{setting}}}\n")));
        assert!(message.contains(field), "{setting}: {message}");
    }
    // the options of the family named
    let cosy =
        tts("extra={family=\"cosyvoice3\",tts_options={top_k=5.0,num_inference_steps=10}}\n");
    assert!(Config::parse(&cosy).is_ok());
}

#[test]
fn speech_synthesis_settings_do_not_belong_to_other_kinds() {
    let music = format!(
        "{BASE}[[model]]\nkey=\"m\"\nbackend=\"audiocpp\"\nkind=\"music\"\nsource={{file=\"/var/lib/eliezer/models/music\"}}\nvram_gb=10.0\nextra={{speed=1.0}}\n"
    );
    assert!(error_of(&music).contains("extra"));
    // a language code stays an error for music, as before
    let music_language = music.replace("speed=1.0", "language=\"de\"");
    assert!(error_of(&music_language).contains("extra"));
}

#[test]
fn the_voice_settings_have_defaults_and_limits() {
    let config = Config::parse(&tts("")).unwrap();
    let audio = config.backend.audiocpp.as_ref().unwrap();
    assert_eq!(
        (
            audio.max_input_chars,
            audio.max_passage_chars,
            audio.pause_ms,
            audio.paragraph_pause_ms
        ),
        (100_000, 400, 400, 800)
    );
    assert_eq!(
        (
            audio.voice_max_file_mb,
            audio.voice_max_seconds,
            audio.voice_name_max_chars,
            audio.voice_sample_rate,
            audio.max_voices
        ),
        (20, 120, 64, 24_000, 100)
    );
    // the voices directory defaults to a directory in the model store, never to a path of a request
    assert_eq!(
        config.voices_dir(),
        PathBuf::from("/var/lib/eliezer/models/voices")
    );
    let set = tts("").replace(
        "port_range=[24700,24709]\n",
        "port_range=[24700,24709]\nvoices_dir=\"/srv/voices\"\n",
    );
    assert_eq!(
        Config::parse(&set).unwrap().voices_dir(),
        PathBuf::from("/srv/voices")
    );
    for (line, field) in [
        ("voices_dir=\"voices\"", "voices_dir"),
        ("voices_dir=\"/srv/../etc\"", "voices_dir"),
        ("max_input_chars=0", "max_input_chars"),
        ("max_input_chars=500000", "max_input_chars"),
        ("max_passage_chars=5", "max_passage_chars"),
        ("pause_ms=20000", "pause_ms"),
        ("paragraph_pause_ms=20000", "pause_ms"),
        ("voice_max_file_mb=0", "voice_max_file_mb"),
        ("voice_max_seconds=0", "voice_max_file_mb"),
        ("voice_name_max_chars=0", "voice_name_max_chars"),
        ("voice_name_max_chars=500", "voice_name_max_chars"),
        ("voice_sample_rate=100", "voice_sample_rate"),
        ("max_voices=0", "max_voices"),
    ] {
        let text = tts("").replace(
            "port_range=[24700,24709]\n",
            &format!("port_range=[24700,24709]\n{line}\n"),
        );
        let message = error_of(&text);
        assert!(message.contains(field), "{line}: {message}");
    }
}

#[test]
fn a_voice_name_is_a_file_name_stem() {
    for name in ["mama", "Papa-2", "a_b", "x", "0"] {
        assert!(valid_voice_name(name, 64), "{name}");
    }
    let long = "a".repeat(65);
    for name in [
        "", "../x", "a/b", "a\\b", ".", "..", ".hidden", "a.wav", "a b", "a\0b", "a\nb", "Müller",
        "名前", "a:b", "~", "$(id)", &long,
    ] {
        assert!(!valid_voice_name(name, 64), "{name:?}");
    }
    assert!(valid_voice_name(&"a".repeat(64), 64));
}
