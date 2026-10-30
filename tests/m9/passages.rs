//! Cutting a text into passages and joining the audio of the passages.
use eliezer_speech::{
    decode_at,
    passages::{Passage, passages},
    pcm::{Pcm, WavError, join, read_wav, write_wav},
    wav_at,
};

fn texts(list: &[Passage]) -> Vec<&str> {
    list.iter().map(|p| p.text.as_str()).collect()
}

#[test]
fn short_text_is_one_passage_and_sentences_are_packed_up_to_the_limit() {
    let out = passages("Es war einmal. Ein Hase lief davon! Wohin denn?", 1000);
    assert_eq!(
        out,
        [Passage {
            text: "Es war einmal. Ein Hase lief davon! Wohin denn?".into(),
            ends_paragraph: true
        }]
    );
    // a limit that holds two sentences but not three
    let out = passages("Eins hier. Zwei hier. Drei hier. Vier hier.", 22);
    assert_eq!(
        texts(&out),
        ["Eins hier. Zwei hier.", "Drei hier. Vier hier."]
    );
    assert!(!out[0].ends_paragraph && out[1].ends_paragraph);
}

#[test]
fn passages_end_at_sentence_boundaries_and_never_exceed_the_limit() {
    let sentence = "Der kleine Bär ging über die Wiese und suchte seinen Honig.";
    let text = vec![sentence; 40].join(" ");
    for max in [60, 100, 400] {
        let out = passages(&text, max);
        assert!(out.len() > 1);
        for p in &out {
            assert!(p.text.chars().count() <= max, "{}", p.text);
            assert!(p.text.ends_with('.'), "{}", p.text);
            assert!(p.text.starts_with("Der kleine"), "{}", p.text);
        }
        // nothing is lost or invented
        let joined: Vec<&str> = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(joined.join(" "), text);
    }
}

#[test]
fn paragraphs_are_kept_apart_and_a_single_line_break_is_a_space() {
    let text =
        "Erster Absatz,\nin zwei Zeilen. Noch ein Satz.\n\n\n  \nZweiter Absatz.\r\n\r\nDritter.";
    let out = passages(text, 1000);
    assert_eq!(
        out,
        [
            Passage {
                text: "Erster Absatz, in zwei Zeilen. Noch ein Satz.".into(),
                ends_paragraph: true
            },
            Passage {
                text: "Zweiter Absatz.".into(),
                ends_paragraph: true
            },
            Passage {
                text: "Dritter.".into(),
                ends_paragraph: true
            },
        ]
    );
}

#[test]
fn numbers_initials_and_lower_case_continuations_do_not_end_a_sentence() {
    let out = passages(
        "Am 3. Mai kam J. R. Tolkien, z. B. mit Hut. Dann ging er.",
        45,
    );
    // the sentence is cut at the real boundary only
    assert_eq!(
        texts(&out),
        [
            "Am 3. Mai kam J. R. Tolkien, z. B. mit Hut.",
            "Dann ging er."
        ]
    );
    // the second is not a boundary either: lower case follows
    let out = passages("Er sagte ca. zehn Stück. Gut.", 25);
    assert_eq!(texts(&out), ["Er sagte ca. zehn Stück.", "Gut."]);
}

#[test]
fn quotes_and_ellipses_stay_with_their_sentence_and_umlauts_survive() {
    let out = passages(
        "„Möchtest du Äpfel?“ fragte sie. „Ja!“ rief er… Schön war’s.",
        40,
    );
    assert_eq!(
        texts(&out),
        [
            "„Möchtest du Äpfel?“ fragte sie.",
            "„Ja!“ rief er… Schön war’s."
        ]
    );
    for p in &out {
        assert!(p.text.chars().count() <= 40);
    }
}

#[test]
fn a_sentence_longer_than_a_passage_is_cut_at_clauses_then_words_then_inside_a_word() {
    let long = "Erst kam der Wind, dann kam der Regen, dann kam der Sturm über die Berge, und alle Tiere liefen nach Hause.";
    let out = passages(long, 40);
    assert!(out.len() >= 3);
    for p in &out {
        assert!(p.text.chars().count() <= 40, "{}", p.text);
    }
    // the clause marks are kept, the first cut is after a comma
    assert_eq!(out[0].text, "Erst kam der Wind, dann kam der Regen,");
    assert_eq!(out.last().unwrap().text.chars().last(), Some('.'));
    // no commas: between words
    let words = vec!["Wort"; 30].join(" ");
    let out = passages(&words, 24);
    assert!(out.iter().all(|p| p.text.chars().count() <= 24));
    assert_eq!(
        out.iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        words
    );
    // one word longer than the limit, with umlauts: cut on character boundaries
    let giant = "Ä".repeat(50);
    let out = passages(&giant, 20);
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|p| p.text.chars().count() <= 20));
    assert_eq!(
        out.iter().map(|p| p.text.as_str()).collect::<String>(),
        giant
    );
}

#[test]
fn empty_text_and_passages_without_letters_give_nothing() {
    assert!(passages("", 100).is_empty());
    assert!(passages("   \n\n \t ", 100).is_empty());
    assert!(passages("* * *\n\n---\n\nEin Satz.", 100).len() == 1);
    assert_eq!(
        passages("* * *\n\n---\n\nEin Satz.", 100)[0].text,
        "Ein Satz."
    );
    // a limit of zero is taken as one
    assert!(!passages("ab", 0).is_empty());
}

#[test]
fn a_whole_chapter_is_cut_in_reasonable_time() {
    let paragraph = "Es war ein Mal ein König, der hatte drei Söhne. Der Älteste war klug, der zweite war stark, der dritte aber war ein Träumer! Wer, fragte er, soll das Reich erben?";
    let text = vec![paragraph; 2000].join("\n\n");
    assert!(text.len() > 300_000);
    let out = passages(&text, 400);
    assert!(out.len() >= 2000);
    assert_eq!(out.iter().filter(|p| p.ends_paragraph).count(), 2000);
    assert!(out.iter().all(|p| p.text.chars().count() <= 400));
}

fn pcm(rate: u32, samples: &[i16]) -> Pcm {
    Pcm {
        rate,
        channels: 1,
        samples: samples.to_vec(),
    }
}

#[test]
fn audio_is_joined_with_the_pauses_between_the_parts_only() {
    let joined = join(vec![
        (pcm(1000, &[1, 2, 3]), 10),
        (pcm(1000, &[4]), 20),
        (pcm(1000, &[5, 6]), 500),
    ])
    .unwrap();
    // 10 ms at 1000 Hz are 10 frames of silence, 20 ms are 20; none after the last part
    let mut expected = vec![1, 2, 3];
    expected.extend([0; 10]);
    expected.push(4);
    expected.extend([0; 20]);
    expected.extend([5, 6]);
    assert_eq!(joined.samples, expected);
    assert_eq!((joined.rate, joined.channels), (1000, 1));
    // a single part is returned as it is
    assert_eq!(join(vec![(pcm(8000, &[7]), 999)]).unwrap().samples, [7]);
    // parts with another rate or channels cannot be joined; nothing to join is an error
    assert_eq!(
        join(vec![(pcm(1000, &[1]), 0), (pcm(2000, &[1]), 0)]),
        Err(WavError::Mismatch)
    );
    let stereo = Pcm {
        rate: 1000,
        channels: 2,
        samples: vec![1, 2],
    };
    assert_eq!(
        join(vec![(pcm(1000, &[1]), 0), (stereo.clone(), 0)]),
        Err(WavError::Mismatch)
    );
    // stereo silence has a sample per channel
    let two = join(vec![(stereo.clone(), 3), (stereo, 0)]).unwrap();
    assert_eq!(two.samples.len(), 2 + 6 + 2);
    assert_eq!(join(vec![]), Err(WavError::Invalid));
}

#[test]
fn wav_files_are_written_and_read_back_and_odd_files_are_refused() {
    let original = pcm(24_000, &[0, 1, -1, 32767, -32768]);
    let bytes = write_wav(&original);
    assert_eq!(bytes.len(), 44 + 10);
    assert_eq!(read_wav(&bytes).unwrap(), original);
    // a streaming writer leaves the data size at 0 or at the maximum: the data run to the end
    for size in [0_u32, u32::MAX] {
        let mut odd = bytes.clone();
        odd[40..44].copy_from_slice(&size.to_le_bytes());
        assert_eq!(read_wav(&odd).unwrap(), original);
    }
    // an extra chunk before the data
    let mut with_list = bytes[..36].to_vec();
    with_list.extend_from_slice(b"LIST\x03\x00\x00\x00abc\x00");
    with_list.extend_from_slice(&bytes[36..]);
    assert_eq!(read_wav(&with_list).unwrap(), original);
    assert_eq!(read_wav(b"not a wav file"), Err(WavError::Invalid));
    assert_eq!(read_wav(&bytes[..30]), Err(WavError::Invalid));
    assert_eq!(read_wav(&bytes[..36]), Err(WavError::Invalid));
    let mut float = bytes.clone();
    float[20] = 3;
    assert_eq!(read_wav(&float), Err(WavError::Unsupported));
    let mut eight = bytes.clone();
    eight[34] = 8;
    assert_eq!(read_wav(&eight), Err(WavError::Unsupported));
}

#[test]
fn a_recording_is_decoded_to_the_requested_rate() {
    // one second of a 440 Hz tone at 44.1 kHz as a WAV file
    let tone: Vec<i16> = (0..44_100)
        .map(|n| ((n as f64 * 440.0 * std::f64::consts::TAU / 44_100.0).sin() * 8000.0) as i16)
        .collect();
    let file = wav_at(&tone, 44_100);
    for rate in [16_000_u32, 24_000, 48_000] {
        let samples = decode_at(file.clone(), 10, rate).unwrap();
        let expected = rate as i64;
        assert!(
            (samples.len() as i64 - expected).abs() < expected / 20,
            "{rate}: {}",
            samples.len()
        );
        assert!(samples.iter().any(|s| s.abs() > 4000));
    }
    // the same file at its own rate is not resampled
    assert_eq!(decode_at(file.clone(), 10, 44_100).unwrap().len(), 44_100);
    // the length limit applies at the requested rate
    assert!(decode_at(file, 0, 24_000).is_err());
}
