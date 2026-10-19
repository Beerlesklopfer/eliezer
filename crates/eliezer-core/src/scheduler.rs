//! State transitions and load planning without side effects.
use std::collections::{BTreeMap, VecDeque};

pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
    Failed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Load,
    LoadDone,
    LoadFailed,
    Unload,
    UnloadDone,
    Crash,
    Retry,
}
impl State {
    pub fn transition(self, event: Event) -> Result<Self, TransitionError> {
        match (self, event) {
            (Self::Unloaded, Event::Load) => Ok(Self::Loading),
            (Self::Loading, Event::LoadDone) => Ok(Self::Loaded),
            (Self::Loading, Event::LoadFailed)
            | (Self::Loaded, Event::Crash)
            | (Self::Unloading, Event::Crash) => Ok(Self::Failed),
            (Self::Loaded, Event::Unload) => Ok(Self::Unloading),
            (Self::Unloading, Event::UnloadDone) | (Self::Failed, Event::Retry) => {
                Ok(Self::Unloaded)
            }
            _ => Err(TransitionError),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionError;
#[derive(Debug, Clone)]
pub struct Instance {
    pub key: String,
    pub state: State,
    pub bytes: u64,
    pub priority: i32,
    pub pinned: bool,
    pub image: bool,
    pub in_flight: usize,
    pub last_used_ms: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    Load,
    LoadAfterUnload(Vec<String>),
    Wait,
    Reject,
}

/// `free` already accounts for the reserve, leases and foreign GPU usage.
pub fn plan_load(instances: &[Instance], request: &Instance, free: u64) -> Plan {
    let exclusive_conflict = |i: &&Instance| {
        (request.image || i.image)
            && i.state != State::Unloaded
            && i.state != State::Failed
            && i.key != request.key
    };
    if instances
        .iter()
        .filter(exclusive_conflict)
        .any(|i| i.pinned)
    {
        return Plan::Reject;
    }
    let conflicts: Vec<_> = instances.iter().filter(exclusive_conflict).collect();
    let eligible = |i: &&Instance| {
        i.key != request.key
            && i.state == State::Loaded
            && !i.pinned
            && (request.image || i.image || i.priority <= request.priority)
    };
    let mut candidates: Vec<_> = instances
        .iter()
        .filter(eligible)
        .filter(|i| i.in_flight == 0)
        .collect();
    candidates.sort_by_key(|i| (i.priority, i.last_used_ms, &i.key));
    let mut reclaimed = free;
    let mut victims = Vec::new();
    for candidate in candidates {
        if reclaimed >= request.bytes && conflicts.iter().all(|i| victims.contains(&i.key)) {
            break;
        }
        reclaimed = reclaimed.saturating_add(candidate.bytes);
        victims.push(candidate.key.clone());
    }
    if reclaimed >= request.bytes && conflicts.iter().all(|i| victims.contains(&i.key)) {
        return if victims.is_empty() {
            Plan::Load
        } else {
            Plan::LoadAfterUnload(victims)
        };
    }
    let mut potential = reclaimed;
    for i in instances
        .iter()
        .filter(eligible)
        .filter(|i| i.in_flight > 0)
    {
        potential = potential.saturating_add(i.bytes);
    }
    let pending = instances.iter().any(|i| i.state == State::Unloading);
    if (potential >= request.bytes
        && instances
            .iter()
            .filter(exclusive_conflict)
            .all(|i| !i.pinned && (i.state == State::Loaded || i.state == State::Unloading)))
        || pending
    {
        Plan::Wait
    } else {
        Plan::Reject
    }
}

pub fn idle_candidates(instances: &[Instance], now_ms: u64, idle_ms: u64) -> Vec<String> {
    instances
        .iter()
        .filter(|i| {
            i.state == State::Loaded
                && i.in_flight == 0
                && !i.pinned
                && now_ms.saturating_sub(i.last_used_ms) >= idle_ms
        })
        .map(|i| i.key.clone())
        .collect()
}

#[derive(Debug)]
pub struct Queued<T> {
    pub id: u64,
    pub enqueued_ms: u64,
    pub value: T,
}
/// FIFO per model, round-robin across models. Batching is requested explicitly via `preferred`.
pub struct Queue<T> {
    models: BTreeMap<String, VecDeque<Queued<T>>>,
    order: VecDeque<String>,
    len: usize,
    max: usize,
}
impl<T> Queue<T> {
    pub fn new(max: usize) -> Self {
        Self {
            models: BTreeMap::new(),
            order: VecDeque::new(),
            len: 0,
            max,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn push(&mut self, model: String, item: Queued<T>) -> Result<(), Queued<T>> {
        if self.len >= self.max {
            return Err(item);
        }
        if !self.models.contains_key(&model) {
            self.order.push_back(model.clone());
        }
        self.models.entry(model).or_default().push_back(item);
        self.len += 1;
        Ok(())
    }
    pub fn pop(&mut self, preferred: Option<&str>) -> Option<(String, Queued<T>)> {
        let index = preferred
            .and_then(|key| self.order.iter().position(|v| v == key))
            .unwrap_or(0);
        let model = self.order.remove(index)?;
        let queue = self.models.get_mut(&model)?;
        let item = queue.pop_front()?;
        if queue.is_empty() {
            self.models.remove(&model);
        } else {
            self.order.push_back(model.clone());
        }
        self.len -= 1;
        Some((model, item))
    }
    /// The supervisor removes disconnected clients and expired jobs here.
    pub fn remove_if(&mut self, mut predicate: impl FnMut(&Queued<T>) -> bool) -> Vec<Queued<T>> {
        let mut removed = Vec::new();
        for queue in self.models.values_mut() {
            let mut retained = VecDeque::new();
            while let Some(item) = queue.pop_front() {
                if predicate(&item) {
                    removed.push(item);
                } else {
                    retained.push_back(item);
                }
            }
            *queue = retained;
        }
        self.models.retain(|_, queue| !queue.is_empty());
        self.order.retain(|key| self.models.contains_key(key));
        self.len -= removed.len();
        removed
    }
}
