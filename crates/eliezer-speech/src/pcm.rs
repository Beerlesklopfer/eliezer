//! Reading the WAV files of a speech synthesis program and joining them with pauses.

/// 16 bit PCM audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcm {
    pub rate: u32,
    pub channels: u16,
    /// Interleaved samples.
    pub samples: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavError {
    /// Not a RIFF/WAVE file, or no `fmt ` and `data` chunk.
    Invalid,
    /// A format other than 16 bit integer PCM.
    Unsupported,
    /// The parts differ in sample rate or channels, so they cannot be joined.
    Mismatch,
}

fn le16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}
fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Reads a WAV file of 16 bit integer PCM. A `data` chunk whose size is wrong or unknown (0 or more than the file
/// holds, as a streaming writer leaves it) is taken to run to the end of the file.
pub fn read_wav(bytes: &[u8]) -> Result<Pcm, WavError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::Invalid);
    }
    let mut at = 12;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = le32(bytes, at + 4).ok_or(WavError::Invalid)? as usize;
        let body = at + 8;
        if id == b"fmt " {
            let tag = le16(bytes, body).ok_or(WavError::Invalid)?;
            let channels = le16(bytes, body + 2).ok_or(WavError::Invalid)?;
            let rate = le32(bytes, body + 4).ok_or(WavError::Invalid)?;
            let bits = le16(bytes, body + 14).ok_or(WavError::Invalid)?;
            format = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) = format.ok_or(WavError::Invalid)?;
            // 0xFFFE is the extensible header; the sample format is then in the sub format, 16 bit integers in practice.
            if !(tag == 1 || tag == 0xFFFE) || bits != 16 {
                return Err(WavError::Unsupported);
            }
            if channels == 0 || rate == 0 {
                return Err(WavError::Invalid);
            }
            let available = bytes.len() - body;
            let length = if size == 0 || size > available {
                available
            } else {
                size
            };
            let samples = bytes[body..body + length]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b))
                .collect();
            return Ok(Pcm {
                rate,
                channels,
                samples,
            });
        }
        // Chunks are padded to an even size.
        at = body.saturating_add(size).saturating_add(size & 1);
    }
    Err(WavError::Invalid)
}

/// A WAV file of the audio.
pub fn write_wav(pcm: &Pcm) -> Vec<u8> {
    let data = (pcm.samples.len() * 2) as u32;
    let block = pcm.channels * 2;
    let mut out = Vec::with_capacity(44 + pcm.samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&pcm.channels.to_le_bytes());
    out.extend_from_slice(&pcm.rate.to_le_bytes());
    out.extend_from_slice(&(pcm.rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for sample in &pcm.samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Joins the audio of the passages: after each part comes a silence of the given milliseconds (none after the last
/// one). All parts must have the same rate and channels.
pub fn join(parts: Vec<(Pcm, u32)>) -> Result<Pcm, WavError> {
    let Some(first) = parts.first() else {
        return Err(WavError::Invalid);
    };
    let (rate, channels) = (first.0.rate, first.0.channels);
    let count = parts.len();
    let mut samples = Vec::new();
    for (index, (pcm, pause_ms)) in parts.into_iter().enumerate() {
        if pcm.rate != rate || pcm.channels != channels {
            return Err(WavError::Mismatch);
        }
        samples.extend_from_slice(&pcm.samples);
        if index + 1 < count && pause_ms > 0 {
            let frames = (u64::from(rate) * u64::from(pause_ms) / 1000) as usize;
            samples.resize(samples.len() + frames * usize::from(channels), 0);
        }
    }
    Ok(Pcm {
        rate,
        channels,
        samples,
    })
}
