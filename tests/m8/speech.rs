use eliezer_backend::{TranscribedSegment, Transcription};
use eliezer_speech::*;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../../tests/fixtures/stt/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// A WAV file with the given format, a sine wave of `seconds`.
fn synth(rate: u32, channels: u16, bits: u16, seconds: f64) -> Vec<u8> {
    let frames = (f64::from(rate) * seconds) as usize;
    let mut data = Vec::new();
    for n in 0..frames {
        let value = (n as f64 * 440.0 * std::f64::consts::TAU / f64::from(rate)).sin() * 0.5;
        for _ in 0..channels {
            match bits {
                8 => data.push(((value * 127.0) as i8 as i16 + 128) as u8),
                _ => data.extend_from_slice(&((value * 32767.0) as i16).to_le_bytes()),
            }
        }
    }
    let block = channels * bits / 8;
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    wav.extend_from_slice(&block.to_le_bytes());
    wav.extend_from_slice(&bits.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    wav
}

fn peak(samples: &[i16]) -> i16 {
    samples
        .iter()
        .map(|s| s.saturating_abs())
        .max()
        .unwrap_or(0)
}

#[test]
fn a_wav_file_at_16_khz_is_taken_as_it_is() {
    let samples = decode(synth(16_000, 1, 16, 0.5), 60).unwrap();
    assert_eq!(samples.len(), 8000);
    assert!(peak(&samples) > 10_000);
}

#[test]
fn stereo_44_khz_is_mixed_down_and_resampled_to_16_khz() {
    let samples = decode(synth(44_100, 2, 16, 2.0), 60).unwrap();
    let seconds = samples.len() as f64 / 16_000.0;
    assert!((1.9..=2.1).contains(&seconds), "{seconds}");
    assert!(peak(&samples) > 10_000, "the tone must survive resampling");
}

#[test]
fn eight_bit_wav_is_decoded() {
    let samples = decode(synth(8_000, 1, 8, 1.0), 60).unwrap();
    assert!(
        (15_000..=17_000).contains(&samples.len()),
        "{}",
        samples.len()
    );
    assert!(peak(&samples) > 5_000);
}

#[test]
fn mp3_flac_vorbis_and_aac_fixtures_are_decoded() {
    for name in ["tone.mp3", "tone.flac", "tone.ogg", "tone.m4a"] {
        let samples = decode(fixture(name), 60).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let seconds = samples.len() as f64 / 16_000.0;
        assert!((0.9..=1.2).contains(&seconds), "{name}: {seconds} s");
        assert!(peak(&samples) > 2_000, "{name}: silent");
    }
}

#[test]
fn something_that_is_not_audio_is_refused() {
    assert_eq!(
        decode(b"This is a text file, not a sermon.".to_vec(), 60),
        Err(DecodeError::NotAudio)
    );
    assert_eq!(decode(Vec::new(), 60), Err(DecodeError::NotAudio));
    assert_eq!(decode(vec![0x5A; 5000], 60), Err(DecodeError::NotAudio));
    // a picture
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[0; 200]);
    assert_eq!(decode(png, 60), Err(DecodeError::NotAudio));
}

#[test]
fn a_recording_over_the_length_limit_is_refused_while_decoding() {
    assert_eq!(
        decode(synth(16_000, 1, 16, 3.0), 2),
        Err(DecodeError::TooLong)
    );
    assert!(decode(synth(16_000, 1, 16, 3.0), 3).is_ok());
    // resampled input too
    assert_eq!(
        decode(synth(44_100, 1, 16, 3.0), 2),
        Err(DecodeError::TooLong)
    );
}

#[test]
fn a_wav_made_from_samples_has_the_header_the_program_wants() {
    let bytes = wav(&[1, -2, 3]);
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..16], b"WAVEfmt ");
    assert_eq!(
        u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        16_000
    );
    assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
    assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
    assert_eq!(bytes.len(), 44 + 6);
    // and our own decoder reads it back
    let again = decode(wav(&vec![1000; 16_000]), 60).unwrap();
    assert_eq!(again.len(), 16_000);
}

#[test]
fn a_short_recording_is_one_piece() {
    assert_eq!(plan(16_000 * 60, 600, 5), vec![0..16_000 * 60]);
    assert_eq!(plan(1, 600, 5), vec![0..1]);
}

#[test]
fn a_long_recording_is_cut_into_overlapping_pieces_that_cover_it() {
    let rate = 16_000;
    let total = rate * 1500; // 25 minutes
    let pieces = plan(total, 600, 5);
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0], 0..rate * 600);
    assert_eq!(
        pieces[1].start,
        rate * 595,
        "the next piece starts 5 s before the end of the last"
    );
    assert_eq!(pieces.last().unwrap().end, total);
    for pair in pieces.windows(2) {
        assert_eq!(pair[0].end - pair[1].start, rate * 5);
    }
    assert!(pieces.iter().all(|p| p.len() <= rate * 600 + rate * 5));
}

#[test]
fn a_tail_that_is_only_overlap_stays_in_the_last_piece() {
    let rate = 16_000;
    let pieces = plan(rate * 602, 600, 5);
    assert_eq!(pieces, vec![0..rate * 602]);
}

fn segment(start: f64, end: f64, text: &str) -> TranscribedSegment {
    TranscribedSegment {
        start,
        end,
        text: text.into(),
    }
}
fn part(offset: f64, segments: Vec<TranscribedSegment>) -> Part {
    Part {
        offset,
        transcription: Transcription {
            language: None,
            segments,
        },
    }
}

#[test]
fn the_join_adds_the_offset_and_drops_what_the_overlap_heard_twice() {
    // pieces of 20 s with 4 s overlap: the second starts at 16 s; the cut is at 18 s
    let first = part(
        0.0,
        vec![
            segment(0.0, 8.0, "one"),
            segment(8.0, 16.5, "two"),
            segment(16.5, 19.5, "three"), // middle 18.0: belongs to the second piece
            segment(19.5, 20.0, "cut off"), // middle 19.75: also the second piece's
        ],
    );
    let second = part(
        16.0,
        vec![
            segment(0.5, 3.5, "three"), // 16.5 .. 19.5, middle 18.0
            segment(3.5, 12.0, "four"),
            segment(0.0, 0.4, "twice"), // 16.0 .. 16.4: before the cut, the first piece has it
        ],
    );
    let joined = join(vec![first, second], 4.0);
    let texts: Vec<_> = joined.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["one", "two", "three", "four"]);
    assert_eq!(joined[2].start, 16.5);
    assert_eq!(joined[2].end, 19.5);
    assert_eq!(joined[3].start, 19.5);
    assert_eq!(joined[3].end, 28.0);
}

#[test]
fn segments_stay_in_order_over_three_pieces_and_empty_ones_vanish() {
    let parts = vec![
        part(0.0, vec![segment(0.0, 5.0, "a"), segment(5.0, 6.0, "  ")]),
        part(10.0, vec![segment(1.0, 4.0, "b")]),
        part(20.0, vec![segment(1.0, 4.0, "c")]),
    ];
    let joined = join(parts, 2.0);
    let texts: Vec<_> = joined.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["a", "b", "c"]);
    assert!(joined.windows(2).all(|w| w[0].start <= w[1].start));
    assert_eq!(joined[2].start, 21.0);
}

#[test]
fn the_text_formats() {
    let segments = vec![
        segment(0.0, 1.5, " Hello "),
        segment(3725.25, 3726.0, "world"),
    ];
    assert_eq!(text(&segments), "Hello world");
    assert_eq!(
        srt(&segments),
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n01:02:05,250 --> 01:02:06,000\nworld\n\n"
    );
    assert_eq!(
        vtt(&segments),
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n\n01:02:05.250 --> 01:02:06.000\nworld\n\n"
    );
    assert_eq!(text(&[]), "");
    assert_eq!(vtt(&[]), "WEBVTT\n\n");
}
