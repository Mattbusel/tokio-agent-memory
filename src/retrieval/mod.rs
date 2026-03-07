use crate::{error::MemoryError, types::MemoryItem};
use chrono::{DateTime, Utc};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};

pub struct RetrievalResult {
    pub item: MemoryItem,
    pub score: f32,
}

pub enum RetrievalMode {
    Exact { key: String },
    Fuzzy { text: String, threshold: f32 },
    Temporal { from: DateTime<Utc>, to: DateTime<Utc> },
    TagIntersection(Vec<String>),
}

pub struct MemoryQuery {
    pub mode: RetrievalMode,
    pub limit: usize,
    pub min_confidence: f32,
}

pub fn retrieve_from_items(
    items: &[MemoryItem],
    query: &MemoryQuery,
) -> Result<Vec<RetrievalResult>, MemoryError> {
    let filtered: Vec<&MemoryItem> = items
        .iter()
        .filter(|i| i.confidence >= query.min_confidence)
        .collect();

    let mut results: Vec<RetrievalResult> = match &query.mode {
        RetrievalMode::Exact { key } => filtered
            .iter()
            .filter(|i| i.tags.iter().any(|t| t == key))
            .map(|i| RetrievalResult { item: (*i).clone(), score: 1.0 })
            .collect(),

        RetrievalMode::Fuzzy { text, threshold } => {
            let matcher = SkimMatcherV2::default();
            filtered
                .iter()
                .filter_map(|i| {
                    let target = format!("{} {}", i.tags.join(" "), i.content.to_string());
                    matcher.fuzzy_match(&target, text).and_then(|score| {
                        let normalized = (score as f32 / 100.0).clamp(0.0, 1.0);
                        if normalized >= *threshold {
                            Some(RetrievalResult { item: (*i).clone(), score: normalized })
                        } else {
                            None
                        }
                    })
                })
                .collect()
        }

        RetrievalMode::Temporal { from, to } => filtered
            .iter()
            .filter(|i| i.created_at >= *from && i.created_at <= *to)
            .map(|i| RetrievalResult { item: (*i).clone(), score: 1.0 })
            .collect(),

        RetrievalMode::TagIntersection(required_tags) => filtered
            .iter()
            .filter(|i| required_tags.iter().all(|t| i.tags.contains(t)))
            .map(|i| RetrievalResult { item: (*i).clone(), score: 1.0 })
            .collect(),
    };

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(query.limit);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;
    use chrono::Duration;

    fn make_item(tags: Vec<&str>, content: &str) -> MemoryItem {
        let mut item = MemoryItem::new(
            serde_json::json!(content),
            tags.iter().map(|s| s.to_string()).collect(),
            AgentId::new("agent"),
            0.5,
        );
        item.confidence = 1.0;
        item
    }

    #[test]
    fn test_exact_hit() {
        let items = vec![make_item(vec!["rust", "systems"], "hello")];
        let query = MemoryQuery {
            mode: RetrievalMode::Exact { key: "rust".into() },
            limit: 10,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_exact_miss_returns_empty() {
        let items = vec![make_item(vec!["python"], "hello")];
        let query = MemoryQuery {
            mode: RetrievalMode::Exact { key: "rust".into() },
            limit: 10,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert!(results.is_empty());
    }

    #[test]
    fn test_temporal_range() {
        let mut item = make_item(vec![], "old");
        item.created_at = Utc::now() - Duration::hours(2);
        let items = vec![item];

        let from = Utc::now() - Duration::hours(3);
        let to = Utc::now() - Duration::hours(1);
        let query = MemoryQuery {
            mode: RetrievalMode::Temporal { from, to },
            limit: 10,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_temporal_excludes_outside_range() {
        let item = make_item(vec![], "now");
        let items = vec![item];

        let future = Utc::now() + Duration::hours(1);
        let query = MemoryQuery {
            mode: RetrievalMode::Temporal { from: future, to: future + Duration::hours(1) },
            limit: 10,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert!(results.is_empty());
    }

    #[test]
    fn test_tag_intersection_all_required() {
        let items = vec![
            make_item(vec!["a", "b"], "both"),
            make_item(vec!["a"], "only a"),
        ];
        let query = MemoryQuery {
            mode: RetrievalMode::TagIntersection(vec!["a".into(), "b".into()]),
            limit: 10,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert_eq!(results.len(), 1);
        assert!(results[0].item.tags.contains(&"b".to_string()));
    }

    #[test]
    fn test_min_confidence_filters() {
        let mut low_conf = make_item(vec!["x"], "low");
        low_conf.confidence = 0.2;
        let items = vec![low_conf];
        let query = MemoryQuery {
            mode: RetrievalMode::Exact { key: "x".into() },
            limit: 10,
            min_confidence: 0.5,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert!(results.is_empty());
    }

    #[test]
    fn test_limit_truncates_results() {
        let items: Vec<MemoryItem> = (0..5)
            .map(|i| make_item(vec!["tag"], &format!("item {i}")))
            .collect();
        let query = MemoryQuery {
            mode: RetrievalMode::TagIntersection(vec!["tag".into()]),
            limit: 3,
            min_confidence: 0.0,
        };
        let results = retrieve_from_items(&items, &query).expect("retrieve");
        assert_eq!(results.len(), 3);
    }
}
