//! One `Backend` in front of three: language and embedding models go to the text runtime, image models to the
//! image runtime and music models to the music runtime. A new instance is routed by the model, a running one by its
//! generation number: the image runtime numbers its instances from `image_base` on and the music runtime from
//! `audio_base` on (the higher base), so nothing has to be remembered per instance.

use crate::{
    Backend, BackendFuture, ChatRequest, Chunk, EmbedRequest, Embeddings, ImageRequest, LoadSpec,
    MusicRequest, Running, Transcription, TranscriptionRequest,
};
use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};
use tokio::sync::mpsc;

pub struct RoutingBackend {
    text: Arc<dyn Backend>,
    image: Arc<dyn Backend>,
    image_base: u64,
    image_models: RwLock<HashSet<String>>,
    audio: Option<(Arc<dyn Backend>, u64)>,
    audio_models: RwLock<HashSet<String>>,
    speech: Option<(Arc<dyn Backend>, u64)>,
    speech_models: RwLock<HashSet<String>>,
}

impl RoutingBackend {
    pub fn new(text: Arc<dyn Backend>, image: Arc<dyn Backend>, image_base: u64) -> Self {
        Self {
            text,
            image,
            image_base,
            image_models: RwLock::new(HashSet::new()),
            audio: None,
            audio_models: RwLock::new(HashSet::new()),
            speech: None,
            speech_models: RwLock::new(HashSet::new()),
        }
    }

    /// Adds the speech recognition runtime; its generation numbers start at `base`, above those of the music runtime.
    pub fn with_speech(mut self, speech: Arc<dyn Backend>, base: u64) -> Self {
        self.speech = Some((speech, base));
        self
    }

    /// The keys of the models the speech recognition runtime serves.
    pub fn set_speech_models(&self, keys: HashSet<String>) {
        match self.speech_models.write() {
            Ok(mut guard) => *guard = keys,
            Err(poisoned) => *poisoned.into_inner() = keys,
        }
    }

    fn is_speech_model(&self, key: &str) -> bool {
        match self.speech_models.read() {
            Ok(guard) => guard.contains(key),
            Err(poisoned) => poisoned.into_inner().contains(key),
        }
    }

    /// Adds the music runtime; its generation numbers start at `base`, above those of the image runtime.
    pub fn with_audio(mut self, audio: Arc<dyn Backend>, base: u64) -> Self {
        self.audio = Some((audio, base));
        self
    }

    /// The keys of the models the music runtime serves.
    pub fn set_audio_models(&self, keys: HashSet<String>) {
        match self.audio_models.write() {
            Ok(mut guard) => *guard = keys,
            Err(poisoned) => *poisoned.into_inner() = keys,
        }
    }

    fn is_audio_model(&self, key: &str) -> bool {
        match self.audio_models.read() {
            Ok(guard) => guard.contains(key),
            Err(poisoned) => poisoned.into_inner().contains(key),
        }
    }

    /// The keys of the models the image runtime serves (at start and on configuration reload).
    pub fn set_image_models(&self, keys: HashSet<String>) {
        match self.image_models.write() {
            Ok(mut guard) => *guard = keys,
            Err(poisoned) => *poisoned.into_inner() = keys,
        }
    }

    fn is_image_model(&self, key: &str) -> bool {
        match self.image_models.read() {
            Ok(guard) => guard.contains(key),
            Err(poisoned) => poisoned.into_inner().contains(key),
        }
    }

    fn of(&self, running: &Running) -> &Arc<dyn Backend> {
        if let Some((speech, base)) = &self.speech
            && running.generation >= *base
        {
            speech
        } else if let Some((audio, base)) = &self.audio
            && running.generation >= *base
        {
            audio
        } else if running.generation >= self.image_base {
            &self.image
        } else {
            &self.text
        }
    }
}

impl Backend for RoutingBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        if let Some((speech, _)) = &self.speech
            && self.is_speech_model(&spec.model)
        {
            speech.start(spec)
        } else if let Some((audio, _)) = &self.audio
            && self.is_audio_model(&spec.model)
        {
            audio.start(spec)
        } else if self.is_image_model(&spec.model) {
            self.image.start(spec)
        } else {
            self.text.start(spec)
        }
    }
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        self.of(running).stop(running)
    }
    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool> {
        self.of(running).health(running)
    }
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, crate::BackendError>>> {
        self.of(running).chat(running, request)
    }
    fn embed<'a>(
        &'a self,
        running: &'a Running,
        request: EmbedRequest,
    ) -> BackendFuture<'a, Embeddings> {
        self.of(running).embed(running, request)
    }
    fn image<'a>(
        &'a self,
        running: &'a Running,
        request: ImageRequest,
    ) -> BackendFuture<'a, Vec<String>> {
        self.of(running).image(running, request)
    }
    fn music<'a>(
        &'a self,
        running: &'a Running,
        request: MusicRequest,
    ) -> BackendFuture<'a, Vec<u8>> {
        self.of(running).music(running, request)
    }
    fn transcribe<'a>(
        &'a self,
        running: &'a Running,
        request: TranscriptionRequest,
    ) -> BackendFuture<'a, Transcription> {
        self.of(running).transcribe(running, request)
    }
}
