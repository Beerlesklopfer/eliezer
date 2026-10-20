//! One `Backend` in front of two: language and embedding models go to the text runtime, image models to the
//! image runtime. A new instance is routed by the model, a running one by its generation number: the image runtime
//! numbers its instances from `image_base` on, so nothing has to be remembered per instance.

use crate::{
    Backend, BackendFuture, ChatRequest, Chunk, EmbedRequest, Embeddings, ImageRequest, LoadSpec,
    Running,
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
}

impl RoutingBackend {
    pub fn new(text: Arc<dyn Backend>, image: Arc<dyn Backend>, image_base: u64) -> Self {
        Self {
            text,
            image,
            image_base,
            image_models: RwLock::new(HashSet::new()),
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
        if running.generation >= self.image_base {
            &self.image
        } else {
            &self.text
        }
    }
}

impl Backend for RoutingBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        if self.is_image_model(&spec.model) {
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
}
