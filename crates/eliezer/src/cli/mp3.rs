//! MP3 encoding of the WAV stream that the service sends (`eliezer speak --format mp3`).
//!
//! The service writes 16 bit PCM WAV with open sizes (`tts.rs`). The header gives the channel count and the sample
//! rate; the samples that follow are encoded block by block with LAME (`mp3lame-encoder`), so the first MP3 bytes
//! are written while the rest of the text is still being read.
use mp3lame_encoder::{
    Bitrate, Builder, Encoder, FlushNoGap, InterleavedPcm, MonoPcm, Quality,
    max_required_buffer_size,
};

/// The size of the WAV header of a stream.
const HEADER: usize = 44;
/// Constant bit rate; 96 kbit/s is plenty for speech at the rates of the speech models.
const BITRATE: Bitrate = Bitrate::Kbps96;

enum State {
    /// The header is not complete yet.
    Header(Vec<u8>),
    Encoding {
        encoder: Box<Encoder>,
        channels: usize,
        /// The first byte of a sample whose second byte is in the next block.
        carry: Option<u8>,
    },
}

pub struct Mp3Encoder {
    state: State,
}

impl Mp3Encoder {
    pub fn new() -> Self {
        Self {
            state: State::Header(Vec::with_capacity(HEADER)),
        }
    }

    /// Encodes one block of the WAV stream and appends the MP3 bytes that are ready to `out`.
    pub fn push(&mut self, block: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
        let mut rest = block;
        if let State::Header(head) = &mut self.state {
            let take = (HEADER - head.len()).min(rest.len());
            head.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if head.len() < HEADER {
                return Ok(());
            }
            let (channels, rate) = parse_header(head)?;
            let mut builder = Builder::new().ok_or("The MP3 encoder cannot be created")?;
            let step =
                |e: mp3lame_encoder::BuildError| format!("The MP3 encoder cannot be set up: {e}");
            builder.set_num_channels(channels as u8).map_err(step)?;
            builder.set_sample_rate(rate).map_err(step)?;
            builder.set_brate(BITRATE).map_err(step)?;
            builder.set_quality(Quality::NearBest).map_err(step)?;
            let encoder = builder.build().map_err(step)?;
            self.state = State::Encoding {
                encoder: Box::new(encoder),
                channels,
                carry: None,
            };
        }
        let State::Encoding {
            encoder,
            channels,
            carry,
        } = &mut self.state
        else {
            return Ok(());
        };
        let mut bytes = Vec::with_capacity(rest.len() + 1);
        bytes.extend(carry.take());
        bytes.extend_from_slice(rest);
        if bytes.len() % 2 == 1 {
            *carry = bytes.pop();
        }
        let samples: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect();
        // Only whole frames (one sample per channel) are encoded; a stereo remainder is not expected from the
        // service, and an odd sample is dropped rather than shifting the channels.
        let usable = samples.len() - samples.len() % *channels;
        let samples = &samples[..usable];
        if samples.is_empty() {
            return Ok(());
        }
        out.reserve(max_required_buffer_size(samples.len()));
        let encoded = if *channels == 1 {
            encoder.encode_to_vec(MonoPcm(samples), out)
        } else {
            encoder.encode_to_vec(InterleavedPcm(samples), out)
        };
        encoded
            .map(|_| ())
            .map_err(|e| format!("MP3 encoding failed: {e}"))
    }

    /// Ends the stream: the last MP3 frames. An empty stream is an error.
    pub fn finish(&mut self, out: &mut Vec<u8>) -> Result<(), String> {
        match &mut self.state {
            State::Header(_) => Err("The service sent no audio".to_owned()),
            State::Encoding { encoder, .. } => {
                out.reserve(7200);
                encoder
                    .flush_to_vec::<FlushNoGap>(out)
                    .map(|_| ())
                    .map_err(|e| format!("MP3 encoding failed: {e}"))
            }
        }
    }
}

/// The channel count and the sample rate of a 16 bit PCM WAV header.
fn parse_header(head: &[u8]) -> Result<(usize, u32), String> {
    let word = |at: usize| u16::from_le_bytes([head[at], head[at + 1]]);
    let long = |at: usize| u32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
    let (format, channels, rate, bits) = (word(20), word(22), long(24), word(34));
    if &head[..4] != b"RIFF" || &head[8..12] != b"WAVE" || format != 1 || bits != 16 {
        return Err(
            "The service sent audio that is not 16 bit PCM WAV; it cannot be made into MP3"
                .to_owned(),
        );
    }
    if !(1..=2).contains(&channels) || rate == 0 {
        return Err(format!(
            "The service sent {channels} channels at {rate} Hz; MP3 takes one or two channels"
        ));
    }
    Ok((usize::from(channels), rate))
}
