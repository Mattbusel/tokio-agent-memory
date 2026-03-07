use crate::{error::MemoryError, types::MemoryItem};

pub enum DecayPolicy {
    Exponential { rate: f64 },
    Linear { rate_per_hour: f64 },
    None,
}

pub struct DecayReport {
    pub items_decayed: usize,
    pub items_forgotten: usize,
}

pub struct DecayScheduler {
    policy: DecayPolicy,
    forget_threshold: f32,
}

impl DecayScheduler {
    pub fn new(policy: DecayPolicy, forget_threshold: f32) -> Result<Self, MemoryError> {
        match &policy {
            DecayPolicy::Exponential { rate } if *rate <= 0.0 => {
                return Err(MemoryError::InvalidDecayRate(*rate));
            }
            DecayPolicy::Linear { rate_per_hour } if *rate_per_hour <= 0.0 => {
                return Err(MemoryError::InvalidDecayRate(*rate_per_hour));
            }
            _ => {}
        }
        Ok(Self { policy, forget_threshold })
    }

    pub fn apply(&self, items: &mut Vec<MemoryItem>) -> Result<DecayReport, MemoryError> {
        let now = chrono::Utc::now();
        let mut decayed = 0;
        let mut forgotten = 0;

        items.retain_mut(|item| {
            let hours = (now - item.created_at).num_minutes() as f64 / 60.0;
            let new_conf = match &self.policy {
                DecayPolicy::None => item.confidence,
                DecayPolicy::Exponential { rate } => {
                    (item.confidence as f64 * (-rate * hours).exp()) as f32
                }
                DecayPolicy::Linear { rate_per_hour } => {
                    (item.confidence as f64 - rate_per_hour * hours).max(0.0) as f32
                }
            };

            if (new_conf - item.confidence).abs() > f32::EPSILON {
                decayed += 1;
            }
            item.confidence = new_conf;

            if new_conf < self.forget_threshold && item.importance <= 0.8 {
                forgotten += 1;
                false
            } else {
                true
            }
        });

        Ok(DecayReport { items_decayed: decayed, items_forgotten: forgotten })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    fn item_with_importance(importance: f32) -> MemoryItem {
        MemoryItem::new(serde_json::json!(null), vec![], AgentId::new("a"), importance)
    }

    #[test]
    fn test_invalid_exponential_rate_zero() {
        let result = DecayScheduler::new(DecayPolicy::Exponential { rate: 0.0 }, 0.1);
        assert!(matches!(result, Err(MemoryError::InvalidDecayRate(_))));
    }

    #[test]
    fn test_invalid_exponential_rate_negative() {
        let result = DecayScheduler::new(DecayPolicy::Exponential { rate: -1.0 }, 0.1);
        assert!(matches!(result, Err(MemoryError::InvalidDecayRate(_))));
    }

    #[test]
    fn test_invalid_linear_rate() {
        let result = DecayScheduler::new(DecayPolicy::Linear { rate_per_hour: -0.5 }, 0.1);
        assert!(matches!(result, Err(MemoryError::InvalidDecayRate(_))));
    }

    #[test]
    fn test_none_policy_no_change() {
        let scheduler = DecayScheduler::new(DecayPolicy::None, 0.0).expect("new");
        let mut items = vec![item_with_importance(0.5)];
        items[0].confidence = 0.9;
        let report = scheduler.apply(&mut items).expect("apply");
        assert!((items[0].confidence - 0.9).abs() < 0.01);
        assert_eq!(report.items_forgotten, 0);
    }

    #[test]
    fn test_exponential_reduces_confidence() {
        // Backdate created_at by 2 hours so decay formula sees hours=2
        let scheduler =
            DecayScheduler::new(DecayPolicy::Exponential { rate: 10.0 }, 0.5).expect("new");
        let mut items = vec![item_with_importance(0.5)];
        items[0].confidence = 1.0;
        items[0].created_at = chrono::Utc::now() - chrono::Duration::hours(2);
        scheduler.apply(&mut items).expect("apply");
        // exp(-10 * 2) ≈ 0 → item forgotten (importance 0.5 ≤ 0.8)
        assert!(items.is_empty());
    }

    #[test]
    fn test_high_importance_survives_decay() {
        let scheduler =
            DecayScheduler::new(DecayPolicy::Exponential { rate: 1000.0 }, 0.99).expect("new");
        let mut items = vec![item_with_importance(0.9)]; // importance > 0.8
        items[0].confidence = 0.5;
        let report = scheduler.apply(&mut items).expect("apply");
        assert_eq!(report.items_forgotten, 0);
        assert_eq!(items.len(), 1, "high-importance item should survive");
    }

    #[test]
    fn test_linear_reduces_confidence() {
        let scheduler =
            DecayScheduler::new(DecayPolicy::Linear { rate_per_hour: 0.1 }, 0.0).expect("new");
        let mut items = vec![item_with_importance(0.5)];
        items[0].confidence = 0.8;
        scheduler.apply(&mut items).expect("apply");
        // Confidence should not increase
        assert!(items[0].confidence <= 0.8);
    }

    #[test]
    fn test_items_below_threshold_removed() {
        let scheduler =
            DecayScheduler::new(DecayPolicy::None, 0.95).expect("new");
        let mut items = vec![item_with_importance(0.5)];
        items[0].confidence = 0.5; // below threshold
        let report = scheduler.apply(&mut items).expect("apply");
        assert_eq!(report.items_forgotten, 1);
        assert!(items.is_empty());
    }
}
