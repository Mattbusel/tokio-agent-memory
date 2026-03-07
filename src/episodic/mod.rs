pub mod event;
pub use event::{CausalChain, Event};

use crate::{error::MemoryError, types::MemoryId};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct EpisodicMemory {
    events: Arc<DashMap<String, Event>>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            events: Arc::new(DashMap::new()),
        }
    }

    pub async fn record(&self, event: Event) -> Result<MemoryId, MemoryError> {
        let id = event.id.clone();
        self.events.insert(id.to_string(), event);
        Ok(id)
    }

    pub async fn recall_sequence(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<Event>, MemoryError> {
        let mut events: Vec<Event> = self
            .events
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .map(|e| e.value().clone())
            .collect();
        events.sort_by_key(|e| e.timestamp);
        Ok(events)
    }

    pub async fn causal_chain(&self, event_id: &MemoryId) -> Result<CausalChain, MemoryError> {
        let root = self
            .events
            .get(&event_id.to_string())
            .map(|e| e.clone())
            .ok_or_else(|| MemoryError::NotFound(event_id.to_string()))?;

        let mut chain = vec![root.clone()];
        let mut current = root;

        // Walk the causal chain backwards (caused_by links)
        while let Some(cause_id) = &current.caused_by {
            match self.events.get(&cause_id.to_string()) {
                Some(e) => {
                    let e = e.clone();
                    chain.push(e.clone());
                    current = e;
                }
                None => break,
            }
        }

        chain.reverse();
        chain.sort_by_key(|e| e.timestamp);
        Ok(chain)
    }

    pub async fn forget(&self, id: &MemoryId) -> Result<(), MemoryError> {
        self.events
            .remove(&id.to_string())
            .map(|_| ())
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_record_and_recall() {
        let mem = EpisodicMemory::new();
        let e = Event::new("something happened", vec!["test".into()]);
        let ts = e.timestamp;
        mem.record(e).await.expect("record");

        let from = ts - Duration::seconds(1);
        let to = ts + Duration::seconds(1);
        let results = mem.recall_sequence(from, to).await.expect("recall");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].description, "something happened");
    }

    #[tokio::test]
    async fn test_recall_empty_range() {
        let mem = EpisodicMemory::new();
        let e = Event::new("old event", vec![]);
        mem.record(e).await.expect("record");

        let future = Utc::now() + Duration::hours(1);
        let results = mem
            .recall_sequence(future, future + Duration::seconds(1))
            .await
            .expect("recall");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_forget_removes_event() {
        let mem = EpisodicMemory::new();
        let e = Event::new("to forget", vec![]);
        let id = mem.record(e).await.expect("record");

        mem.forget(&id).await.expect("forget");

        let ts = Utc::now();
        let results = mem
            .recall_sequence(ts - Duration::hours(1), ts + Duration::hours(1))
            .await
            .expect("recall");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_forget_missing_returns_error() {
        let mem = EpisodicMemory::new();
        let fake = MemoryId::new();
        let result = mem.forget(&fake).await;
        assert!(matches!(result, Err(MemoryError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_causal_chain_single_event() {
        let mem = EpisodicMemory::new();
        let e = Event::new("root", vec![]);
        let id = e.id.clone();
        mem.record(e).await.expect("record");

        let chain = mem.causal_chain(&id).await.expect("chain");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].description, "root");
    }

    #[tokio::test]
    async fn test_causal_chain_linked_events() {
        let mem = EpisodicMemory::new();
        let cause = Event::new("cause", vec![]);
        let cause_id = cause.id.clone();
        mem.record(cause).await.expect("record");

        let effect = Event::new("effect", vec![]).with_cause(cause_id.clone());
        let effect_id = effect.id.clone();
        mem.record(effect).await.expect("record");

        let chain = mem.causal_chain(&effect_id).await.expect("chain");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].id, cause_id);
    }

    #[tokio::test]
    async fn test_causal_chain_missing_root_returns_error() {
        let mem = EpisodicMemory::new();
        let fake = MemoryId::new();
        let result = mem.causal_chain(&fake).await;
        assert!(matches!(result, Err(MemoryError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_recall_multiple_events_ordered() {
        let mem = EpisodicMemory::new();
        for i in 0..5 {
            let mut e = Event::new(format!("event {i}"), vec![]);
            e.timestamp = Utc::now() + Duration::seconds(i);
            mem.record(e).await.expect("record");
        }
        let from = Utc::now() - Duration::seconds(1);
        let to = Utc::now() + Duration::seconds(10);
        let results = mem.recall_sequence(from, to).await.expect("recall");
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].timestamp <= w[1].timestamp);
        }
    }
}
