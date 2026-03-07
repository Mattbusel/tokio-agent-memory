use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryItem {
    pub id: MemoryId,
    pub content: serde_json::Value,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub confidence: f32,
    pub importance: f32,
    pub source_agent: AgentId,
}

impl MemoryItem {
    pub fn new(
        content: serde_json::Value,
        tags: Vec<String>,
        agent: AgentId,
        importance: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: MemoryId::new(),
            content,
            tags,
            created_at: now,
            last_accessed: now,
            confidence: 1.0,
            importance,
            source_agent: agent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_uniqueness() {
        let a = MemoryId::new();
        let b = MemoryId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_memory_id_default_unique() {
        let a = MemoryId::default();
        let b = MemoryId::default();
        assert_ne!(a, b);
    }

    #[test]
    fn test_memory_item_serialization_round_trip() {
        let agent = AgentId::new("agent-1");
        let item = MemoryItem::new(
            serde_json::json!({"key": "value", "num": 42}),
            vec!["tag1".into(), "tag2".into()],
            agent.clone(),
            0.8,
        );
        let json = serde_json::to_string(&item).expect("serialize");
        let restored: MemoryItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.id, item.id);
        assert_eq!(restored.tags, item.tags);
        assert_eq!(restored.source_agent, agent);
        assert!((restored.confidence - 1.0).abs() < f32::EPSILON);
        assert!((restored.importance - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_item_constructor_sets_timestamps() {
        let before = Utc::now();
        let item = MemoryItem::new(
            serde_json::json!(null),
            vec![],
            AgentId::new("a"),
            0.5,
        );
        let after = Utc::now();
        assert!(item.created_at >= before);
        assert!(item.created_at <= after);
        assert_eq!(item.created_at, item.last_accessed);
    }

    #[test]
    fn test_agent_id_display() {
        let a = AgentId::new("my-agent");
        assert_eq!(a.to_string(), "my-agent");
    }
}
