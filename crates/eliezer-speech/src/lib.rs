//! What the service does with the audio of a transcription request before and after the speech program sees it:
//! decode any uploaded file to 16 kHz mono 16 bit samples (pure Rust: symphonia decodes, rubato resamples), cut a long
//! recording into overlapping pieces, wrap a piece as a WAV file, join the transcripts of the pieces by time and
//! write them as plain text, SRT or WebVTT.

use eliezer_backend::{TranscribedSegment, Transcription};
use rubato::{FftFixedIn, Resampler};
use std::ops::Range;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

/// The sample rate the speech program wants.
pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// No format or codec of the build recognizes the file, or it holds no audio.
    NotAudio,
    /// The recording is longer than the limit.
    TooLong,
}

/// Decodes a file of any supported format (MP3, AAC in MP4/M4A, Ogg Vorbis, FLAC, WAV) to mono samples at
/// 16 kHz. Decoding stops with [`DecodeError::TooLong`] as soon as the result would be longer than `max_seconds`, so
/// a hostile file cannot fill the memory.
pub fn decode(bytes: Vec<u8>, max_seconds: u64) -> Result<Vec<i16>, DecodeError> {
    let source = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|_| DecodeError::NotAudio)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(DecodeError::NotAudio)?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|_| DecodeError::NotAudio)?;
    let limit =
        usize::try_from(max_seconds.saturating_mul(u64::from(SAMPLE_RATE))).unwrap_or(usize::MAX);
    let mut out: Vec<i16> = Vec::new();
    let mut resampler: Option<Resampling> = None;
    let mut buffer: Option<SampleBuffer<f32>> = None;
    // The end of the stream, or a stream that cannot go on: what was decoded so far is the recording.
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // One damaged packet: skip it.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        let source_rate = if rate > 0 { rate } else { spec.rate };
        if channels == 0 || source_rate == 0 {
            return Err(DecodeError::NotAudio);
        }
        let capacity = decoded.capacity() as u64;
        if buffer
            .as_ref()
            .is_none_or(|b| (b.capacity() as u64) < capacity * channels as u64)
        {
            buffer = Some(SampleBuffer::<f32>::new(capacity, spec));
        }
        let Some(samples) = buffer.as_mut() else {
            return Err(DecodeError::NotAudio);
        };
        samples.copy_interleaved_ref(decoded);
        let mono: Vec<f32> = samples
            .samples()
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        if source_rate == SAMPLE_RATE {
            push(&mut out, &mono);
        } else {
            if resampler.is_none() {
                resampler = Some(Resampling::new(source_rate).ok_or(DecodeError::NotAudio)?);
            }
            if let Some(r) = resampler.as_mut() {
                r.feed(&mono, &mut out);
            }
        }
        if out.len() > limit {
            return Err(DecodeError::TooLong);
        }
    }
    if let Some(r) = resampler.as_mut() {
        r.finish(&mut out);
    }
    if out.len() > limit {
        return Err(DecodeError::TooLong);
    }
    if out.is_empty() {
        return Err(DecodeError::NotAudio);
    }
    Ok(out)
}

fn push(out: &mut Vec<i16>, samples: &[f32]) {
    out.extend(
        samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16),
    );
}

/// A streaming resampler to 16 kHz for one channel.
struct Resampling {
    inner: FftFixedIn<f32>,
    pending: Vec<f32>,
}
impl Resampling {
    fn new(rate: u32) -> Option<Self> {
        let inner = FftFixedIn::<f32>::new(rate as usize, SAMPLE_RATE as usize, 1024, 2, 1).ok()?;
        Some(Self {
            inner,
            pending: Vec::new(),
        })
    }
    fn feed(&mut self, samples: &[f32], out: &mut Vec<i16>) {
        self.pending.extend_from_slice(samples);
        loop {
            let need = self.inner.input_frames_next();
            if self.pending.len() < need {
                break;
            }
            let block: Vec<f32> = self.pending.drain(..need).collect();
            if let Ok(done) = self.inner.process(&[block], None) {
                push(out, &done[0]);
            } else {
                break;
            }
        }
    }
    fn finish(&mut self, out: &mut Vec<i16>) {
        if self.pending.is_empty() {
            return;
        }
        let rest = std::mem::take(&mut self.pending);
        if let Ok(done) = self.inner.process_partial(Some(&[rest]), None) {
            push(out, &done[0]);
        }
    }
}

/// A WAV file (16 kHz, mono, 16 bit) of the samples.
pub fn wav(samples: &[i16]) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2_u16.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// The pieces a recording of `total` samples is cut into: each at most `chunk_secs` long (the last one can be a
/// little longer, so that no piece is only overlap), each starting `chunk_secs - overlap_secs` after the one before.
pub fn plan(total: usize, chunk_secs: u32, overlap_secs: u32) -> Vec<Range<usize>> {
    let rate = SAMPLE_RATE as usize;
    let chunk = (chunk_secs as usize * rate).max(1);
    let overlap = (overlap_secs as usize * rate).min(chunk / 2);
    let step = (chunk - overlap).max(1);
    let mut pieces = Vec::new();
    let mut start = 0;
    loop {
        let mut end = (start + chunk).min(total);
        if total - end <= overlap {
            end = total;
        }
        pieces.push(start..end);
        if end == total {
            break;
        }
        start += step;
    }
    pieces
}

/// The transcript of one piece and where the piece starts in the recording (seconds).
pub struct Part {
    pub offset: f64,
    pub transcription: Transcription,
}

/// Joins the transcripts of the pieces. Times become relative to the whole recording. In the overlap of two pieces both
/// have heard the same speech: the cut lies in the middle of the overlap, and a segment counts for the piece
/// on whose side its middle is. Segments without text are dropped.
pub fn join(parts: Vec<Part>, overlap_secs: f64) -> Vec<TranscribedSegment> {
    let cuts: Vec<f64> = parts
        .iter()
        .skip(1)
        .map(|part| part.offset + overlap_secs / 2.0)
        .collect();
    let mut out = Vec::new();
    for (index, part) in parts.into_iter().enumerate() {
        let lower = if index == 0 {
            f64::NEG_INFINITY
        } else {
            cuts[index - 1]
        };
        let upper = cuts.get(index).copied().unwrap_or(f64::INFINITY);
        for segment in part.transcription.segments {
            let (start, end) = (segment.start + part.offset, segment.end + part.offset);
            let middle = (start + end) / 2.0;
            if middle >= lower && middle < upper && !segment.text.trim().is_empty() {
                out.push(TranscribedSegment {
                    start,
                    end,
                    text: segment.text,
                });
            }
        }
    }
    out
}

/// The whole text: the segments with one space between them.
pub fn text(segments: &[TranscribedSegment]) -> String {
    segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

fn stamp(seconds: f64, separator: char) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        millis / 3_600_000,
        millis / 60_000 % 60,
        millis / 1000 % 60,
        millis % 1000
    )
}

/// SubRip: numbered cues with `HH:MM:SS,mmm`.
pub fn srt(segments: &[TranscribedSegment]) -> String {
    let mut out = String::new();
    for (index, s) in segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            index + 1,
            stamp(s.start, ','),
            stamp(s.end, ','),
            s.text.trim()
        ));
    }
    out
}

/// WebVTT: a header and cues with `HH:MM:SS.mmm`.
pub fn vtt(segments: &[TranscribedSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            stamp(s.start, '.'),
            stamp(s.end, '.'),
            s.text.trim()
        ));
    }
    out
}
