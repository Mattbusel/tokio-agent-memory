pub mod concept;
pub use concept::{Concept, ConceptGraph, Fact, FactId, Relation};

use crate::error::MemoryError;
use dashmap::DashMap;
use std::sync::{atomic::{AtomicU64, Ordering}, Arc};

pub struct SemanticMemory {
    facts: Arc<DashMap<u64, Fact>>,
    next_id: Arc<AtomicU64>,
}

impl SemanticMemory {
    pub fn new() -> Self {
        Self {
            facts: Arc::new(DashMap::new()),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub async fn assert_fact(
        &self,
        subject: Concept,
        relation: Relation,
        object: Concept,
    ) -> Result<FactId, MemoryError> {
        let id = FactId(self.next_id.fetch_add(1, Ordering::SeqCst));
        self.facts.insert(id.0, Fact { id, subject, relation, object });
        Ok(id)
    }

    pub async fn query(&self, concept: &Concept) -> Result<Vec<(Relation, Concept)>, MemoryError> {
        let results = self
            .facts
            .iter()
            .filter(|f| &f.subject == concept)
            .map(|f| (f.relation.clone(), f.object.clone()))
            .collect();
        Ok(results)
    }

    pub async fn related(
        &self,
        concept: &Concept,
        depth: usize,
    ) -> Result<ConceptGraph, MemoryError> {
        let mut edges = Vec::new();
        let mut frontier = vec![concept.clone()];

        for _ in 0..depth {
            let mut next_frontier = Vec::new();
            for c in &frontier {
                for f in self.facts.iter().filter(|f| &f.subject == c) {
                    edges.push((f.relation.clone(), f.object.clone()));
                    next_frontier.push(f.object.clone());
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        Ok(ConceptGraph {
            root: concept.clone(),
            edges,
        })
    }

    pub async fn retract(&self, fact: FactId) -> Result<(), MemoryError> {
        self.facts
            .remove(&fact.0)
            .map(|_| ())
            .ok_or_else(|| MemoryError::NotFound(format!("FactId({})", fact.0)))
    }
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_assert_and_query() {
        let mem = SemanticMemory::new();
        mem.assert_fact(
            Concept::new("Rust"),
            Relation::new("is_a"),
            Concept::new("language"),
        )
        .await
        .expect("assert");

        let results = mem.query(&Concept::new("Rust")).await.expect("query");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, Relation::new("is_a"));
        assert_eq!(results[0].1, Concept::new("language"));
    }

    #[tokio::test]
    async fn test_query_unknown_concept_returns_empty() {
        let mem = SemanticMemory::new();
        let results = mem.query(&Concept::new("unknown")).await.expect("query");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_retract_removes_fact() {
        let mem = SemanticMemory::new();
        let id = mem
            .assert_fact(Concept::new("A"), Relation::new("rel"), Concept::new("B"))
            .await
            .expect("assert");

        mem.retract(id).await.expect("retract");
        let results = mem.query(&Concept::new("A")).await.expect("query");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_retract_missing_returns_error() {
        let mem = SemanticMemory::new();
        let result = mem.retract(FactId(9999)).await;
        assert!(matches!(result, Err(MemoryError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_related_depth_1() {
        let mem = SemanticMemory::new();
        mem.assert_fact(Concept::new("A"), Relation::new("rel"), Concept::new("B"))
            .await
            .expect("assert");
        mem.assert_fact(Concept::new("B"), Relation::new("rel2"), Concept::new("C"))
            .await
            .expect("assert");

        let graph = mem.related(&Concept::new("A"), 1).await.expect("related");
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].1, Concept::new("B"));
    }

    #[tokio::test]
    async fn test_related_depth_2() {
        let mem = SemanticMemory::new();
        mem.assert_fact(Concept::new("A"), Relation::new("r1"), Concept::new("B"))
            .await
            .expect("assert");
        mem.assert_fact(Concept::new("B"), Relation::new("r2"), Concept::new("C"))
            .await
            .expect("assert");

        let graph = mem.related(&Concept::new("A"), 2).await.expect("related");
        assert_eq!(graph.edges.len(), 2);
    }

    #[tokio::test]
    async fn test_multiple_facts_same_subject() {
        let mem = SemanticMemory::new();
        mem.assert_fact(Concept::new("X"), Relation::new("r1"), Concept::new("Y"))
            .await
            .expect("assert");
        mem.assert_fact(Concept::new("X"), Relation::new("r2"), Concept::new("Z"))
            .await
            .expect("assert");

        let results = mem.query(&Concept::new("X")).await.expect("query");
        assert_eq!(results.len(), 2);
    }
}
