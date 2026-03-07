use crate::{
    episodic::{EpisodicMemory, Event},
    error::MemoryError,
    semantic::{Concept, Relation, SemanticMemory},
    working::WorkingMemory,
};

pub struct ConsolidationPolicy {
    pub min_importance_for_semantic: f32,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self { min_importance_for_semantic: 0.8 }
    }
}

pub struct ConsolidationReport {
    pub promoted_to_episodic: usize,
    pub promoted_to_semantic: usize,
}

pub struct Consolidator {
    pub policy: ConsolidationPolicy,
}

impl Consolidator {
    pub fn new(policy: ConsolidationPolicy) -> Self {
        Self { policy }
    }

    pub async fn run(
        &self,
        working: &WorkingMemory,
        episodic: &EpisodicMemory,
        semantic: &SemanticMemory,
    ) -> Result<ConsolidationReport, MemoryError> {
        let items = working.active();
        let mut promoted_to_episodic = 0;
        let mut promoted_to_semantic = 0;

        for item in &items {
            if item.importance >= self.policy.min_importance_for_semantic {
                // High-importance items go to semantic memory as concepts
                let concept = Concept::new(item.content.to_string());
                let relation = Relation::new("is_known");
                let object = Concept::new(format!("importance:{:.2}", item.importance));
                semantic.assert_fact(concept, relation, object).await?;
                promoted_to_semantic += 1;
            } else {
                // Normal items go to episodic memory
                let event = Event::new(
                    format!("consolidated: {}", item.content),
                    item.tags.clone(),
                );
                episodic.record(event).await?;
                promoted_to_episodic += 1;
            }
        }

        Ok(ConsolidationReport { promoted_to_episodic, promoted_to_semantic })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{types::AgentId, working::SlotPriority};
    use crate::types::MemoryItem;

    fn make_item(importance: f32) -> MemoryItem {
        MemoryItem::new(serde_json::json!("data"), vec![], AgentId::new("a"), importance)
    }

    #[tokio::test]
    async fn test_high_importance_promoted_to_semantic() {
        let wm = WorkingMemory::new(5).expect("wm");
        let episodic = EpisodicMemory::new();
        let semantic = SemanticMemory::new();
        let consolidator = Consolidator::new(ConsolidationPolicy { min_importance_for_semantic: 0.8 });

        wm.focus(make_item(0.9), SlotPriority::High).expect("focus");
        let report = consolidator.run(&wm, &episodic, &semantic).await.expect("run");

        assert_eq!(report.promoted_to_semantic, 1);
        assert_eq!(report.promoted_to_episodic, 0);
    }

    #[tokio::test]
    async fn test_normal_item_promoted_to_episodic() {
        let wm = WorkingMemory::new(5).expect("wm");
        let episodic = EpisodicMemory::new();
        let semantic = SemanticMemory::new();
        let consolidator = Consolidator::new(ConsolidationPolicy { min_importance_for_semantic: 0.8 });

        wm.focus(make_item(0.4), SlotPriority::Normal).expect("focus");
        let report = consolidator.run(&wm, &episodic, &semantic).await.expect("run");

        assert_eq!(report.promoted_to_episodic, 1);
        assert_eq!(report.promoted_to_semantic, 0);
    }

    #[tokio::test]
    async fn test_mixed_items_consolidated() {
        let wm = WorkingMemory::new(5).expect("wm");
        let episodic = EpisodicMemory::new();
        let semantic = SemanticMemory::new();
        let consolidator = Consolidator::new(ConsolidationPolicy::default());

        wm.focus(make_item(0.9), SlotPriority::High).expect("high");
        wm.focus(make_item(0.3), SlotPriority::Low).expect("low");
        let report = consolidator.run(&wm, &episodic, &semantic).await.expect("run");

        assert_eq!(report.promoted_to_semantic, 1);
        assert_eq!(report.promoted_to_episodic, 1);
    }

    #[tokio::test]
    async fn test_empty_working_memory_no_report() {
        let wm = WorkingMemory::new(5).expect("wm");
        let episodic = EpisodicMemory::new();
        let semantic = SemanticMemory::new();
        let consolidator = Consolidator::new(ConsolidationPolicy::default());

        let report = consolidator.run(&wm, &episodic, &semantic).await.expect("run");
        assert_eq!(report.promoted_to_episodic, 0);
        assert_eq!(report.promoted_to_semantic, 0);
    }
}
