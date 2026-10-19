//! Conservative VRAM calculation without file system or GPU access.
use crate::config::{Architecture, ModelConfig, ModelKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryError {
    InvalidSize,
    MissingWeights,
    InvalidArchitecture,
    Overflow,
}
impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidSize => "Invalid memory size or load parameters",
            Self::MissingWeights => "Weight size missing",
            Self::InvalidArchitecture => "Invalid architecture",
            Self::Overflow => "Memory calculation exceeds the numeric range",
        })
    }
}
impl std::error::Error for MemoryError {}

pub fn gb_bytes(gb: f64) -> Result<u64, MemoryError> {
    let bytes = gb * 1_000_000_000.0;
    if !bytes.is_finite() || bytes < 0.0 || bytes >= u64::MAX as f64 {
        return Err(MemoryError::InvalidSize);
    }
    Ok(bytes.ceil() as u64)
}
fn product(values: &[u64]) -> Result<u64, MemoryError> {
    values
        .iter()
        .try_fold(1_u64, |a, b| a.checked_mul(*b).ok_or(MemoryError::Overflow))
}
fn sum(values: &[u64]) -> Result<u64, MemoryError> {
    values
        .iter()
        .try_fold(0_u64, |a, b| a.checked_add(*b).ok_or(MemoryError::Overflow))
}

pub fn cache_bytes(arch: &Architecture, context: u32, parallel: u32) -> Result<u64, MemoryError> {
    if context == 0 || parallel == 0 || parallel > context {
        return Err(MemoryError::InvalidSize);
    }
    let swa = arch.swa_layers.unwrap_or(0);
    let linear = arch.linear_layers.unwrap_or(0);
    let full = arch
        .layers
        .checked_sub(swa)
        .and_then(|v| v.checked_sub(linear))
        .ok_or(MemoryError::InvalidArchitecture)?;
    if arch.layers == 0
        || arch.kv_heads == 0
        || arch.head_dim == 0
        || arch.full_kv_heads == Some(0)
        || arch.full_head_dim == Some(0)
    {
        return Err(MemoryError::InvalidArchitecture);
    }
    let full_cache = product(&[
        4,
        full.into(),
        arch.full_kv_heads.unwrap_or(arch.kv_heads).into(),
        arch.full_head_dim.unwrap_or(arch.head_dim).into(),
        context.into(),
    ])?;
    let window_cache = if swa == 0 {
        0
    } else {
        let window = arch
            .window
            .filter(|v| *v > 0)
            .ok_or(MemoryError::InvalidArchitecture)?;
        let tokens = (u64::from(window) * u64::from(parallel) + 2048).min(context.into());
        product(&[
            4,
            swa.into(),
            arch.kv_heads.into(),
            arch.head_dim.into(),
            tokens,
        ])?
    };
    let state = if linear == 0 {
        0
    } else {
        product(&[
            linear.into(),
            parallel.into(),
            arch.state_bytes
                .filter(|v| *v > 0)
                .ok_or(MemoryError::InvalidArchitecture)?,
        ])?
    };
    sum(&[full_cache, window_cache, state])
}

/// Where a memory figure comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Measured in operation.
    Measured,
    /// Fixed by `vram_gb` in the configuration.
    Configured,
    /// Calculated from weights, architecture, context and parallelism.
    Calculated,
}
impl Basis {
    pub fn name(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Configured => "configured",
            Self::Calculated => "calculated",
        }
    }
}

/// An estimate with its parts; for measured or configured values only `total` is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Breakdown {
    pub total: u64,
    pub weights: u64,
    pub cache: u64,
    pub overhead: u64,
    pub vision: u64,
    pub basis: Basis,
}

/// Measured values describe the whole instance; file sizes are supplied by the caller.
pub fn estimate_parts(
    model: &ModelConfig,
    context: u32,
    parallel: u32,
    file_size: Option<u64>,
    measured: Option<u64>,
) -> Result<Breakdown, MemoryError> {
    if context == 0 || context > model.max_context || parallel == 0 || parallel > context {
        return Err(MemoryError::InvalidSize);
    }
    let whole = |total, basis| Breakdown {
        total,
        weights: 0,
        cache: 0,
        overhead: 0,
        vision: 0,
        basis,
    };
    if let Some(bytes) = measured {
        return Ok(whole(bytes, Basis::Measured));
    }
    if let Some(gb) = model.vram_gb {
        return Ok(whole(gb_bytes(gb)?, Basis::Configured));
    }
    if model.kind == ModelKind::Image {
        return Err(MemoryError::MissingWeights);
    }
    let weights = match model.size_gb {
        Some(gb) => gb_bytes(gb)?,
        None => file_size.ok_or(MemoryError::MissingWeights)?,
    };
    let fallback = Architecture {
        layers: 80,
        kv_heads: 8,
        head_dim: 128,
        swa_layers: None,
        window: None,
        full_kv_heads: None,
        full_head_dim: None,
        linear_layers: None,
        state_bytes: None,
    };
    let cache = if model.kind == ModelKind::Embedding {
        0
    } else {
        cache_bytes(model.arch.as_ref().unwrap_or(&fallback), context, parallel)?
    };
    let overhead = if model.kind == ModelKind::Embedding {
        250_000_000
    } else {
        500_000_000
    };
    let vision = if model.vision { 300_000_000 } else { 0 };
    Ok(Breakdown {
        total: sum(&[weights, cache, overhead, vision])?,
        weights,
        cache,
        overhead,
        vision,
        basis: Basis::Calculated,
    })
}

pub fn estimate(
    model: &ModelConfig,
    context: u32,
    parallel: u32,
    file_size: Option<u64>,
    measured: Option<u64>,
) -> Result<u64, MemoryError> {
    estimate_parts(model, context, parallel, file_size, measured).map(|b| b.total)
}

/// The largest context (at most the model's maximum) whose estimate fits into `budget`, or `None`
/// when even the smallest context does not fit. The estimate grows with the context, so this is a
/// binary search.
pub fn max_context_within(
    model: &ModelConfig,
    parallel: u32,
    file_size: Option<u64>,
    measured: Option<u64>,
    budget: u64,
) -> Option<u32> {
    let fits = |context: u32| {
        estimate(model, context, parallel, file_size, measured).is_ok_and(|bytes| bytes <= budget)
    };
    let lowest = parallel.max(1);
    if !fits(lowest) {
        return None;
    }
    let (mut low, mut high) = (lowest, model.max_context);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if fits(middle) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Some(low)
}

/// Foreign usage and own leases are not subtracted twice.
pub fn available(budget: u64, reserve: u64, total: u64, used: u64, leases: u64, own: u64) -> u64 {
    budget
        .min(total)
        .saturating_sub(reserve)
        .saturating_sub(leases.max(used.saturating_sub(own)))
}
