//! HTTP-Programmrahmen; Modellverwaltung und Authentifizierung folgen in M3.
use axum::{Router, routing::get};

pub fn router() -> Router {
    Router::new().route("/healthz", get(|| async { "ok" }))
}
