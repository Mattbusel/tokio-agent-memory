use crate::types::MemoryId;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Event {
    pub id: MemoryId,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub tags: Vec<String>,
    pub caused_by: Option<MemoryId>,
}

impl Event {
    pub fn new(description: impl Into<String>, tags: Vec<String>) -> Self {
        Self {
            id: MemoryId::new(),
            timestamp: Utc::now(),
            description: description.into(),
            tags,
            caused_by: None,
        }
    }

    pub fn with_cause(mut self, cause: MemoryId) -> Self {
        self.caused_by = Some(cause);
        self
    }
}

pub type CausalChain = Vec<Event>;
