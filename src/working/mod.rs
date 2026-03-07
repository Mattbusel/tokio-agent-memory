use crate::{error::MemoryError, types::{MemoryId, MemoryItem}};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlotPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

struct Slot {
    item: MemoryItem,
    priority: SlotPriority,
    inserted_at: std::time::Instant,
}

pub struct WorkingMemory {
    capacity: usize,
    slots: Arc<RwLock<Vec<Slot>>>,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Result<Self, MemoryError> {
        if capacity == 0 {
            return Err(MemoryError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            slots: Arc::new(RwLock::new(Vec::with_capacity(capacity))),
        })
    }

    pub fn focus(&self, item: MemoryItem, priority: SlotPriority) -> Result<(), MemoryError> {
        let mut slots = self.slots.write();
        if slots.len() < self.capacity {
            slots.push(Slot { item, priority, inserted_at: std::time::Instant::now() });
            return Ok(());
        }
        // Find lowest-priority, oldest slot to evict
        let evict_idx = slots
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then(a.inserted_at.cmp(&b.inserted_at))
            })
            .map(|(i, _)| i)
            .ok_or(MemoryError::CapacityExceeded(self.capacity))?;

        slots[evict_idx] = Slot { item, priority, inserted_at: std::time::Instant::now() };
        Ok(())
    }

    pub fn active(&self) -> Vec<MemoryItem> {
        self.slots.read().iter().map(|s| s.item.clone()).collect()
    }

    pub fn clear_slot(&self, id: &MemoryId) -> Result<(), MemoryError> {
        let mut slots = self.slots.write();
        let pos = slots
            .iter()
            .position(|s| &s.item.id == id)
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
        slots.remove(pos);
        Ok(())
    }

    pub fn utilization(&self) -> f32 {
        let len = self.slots.read().len();
        len as f32 / self.capacity as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    fn item(importance: f32) -> MemoryItem {
        MemoryItem::new(
            serde_json::json!(null),
            vec![],
            AgentId::new("agent"),
            importance,
        )
    }

    #[test]
    fn test_zero_capacity_returns_error() {
        let result = WorkingMemory::new(0);
        assert!(matches!(result, Err(MemoryError::ZeroCapacity)));
    }

    #[test]
    fn test_focus_fills_slots() {
        let wm = WorkingMemory::new(3).expect("new");
        wm.focus(item(0.5), SlotPriority::Normal).expect("focus 1");
        wm.focus(item(0.5), SlotPriority::Normal).expect("focus 2");
        assert!((wm.utilization() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_focus_evicts_lowest_priority() {
        let wm = WorkingMemory::new(2).expect("new");
        let low = item(0.1);
        let low_id = low.id.clone();
        wm.focus(low, SlotPriority::Low).expect("low");
        wm.focus(item(0.5), SlotPriority::High).expect("high");
        // Now at capacity; adding Critical should evict Low
        wm.focus(item(0.9), SlotPriority::Critical).expect("critical");

        let active = wm.active();
        assert_eq!(active.len(), 2);
        assert!(!active.iter().any(|i| i.id == low_id));
    }

    #[test]
    fn test_clear_slot_removes_item() {
        let wm = WorkingMemory::new(3).expect("new");
        let it = item(0.5);
        let id = it.id.clone();
        wm.focus(it, SlotPriority::Normal).expect("focus");
        wm.clear_slot(&id).expect("clear");
        assert!(wm.active().is_empty());
    }

    #[test]
    fn test_clear_slot_missing_returns_error() {
        let wm = WorkingMemory::new(3).expect("new");
        let fake = MemoryId::new();
        let result = wm.clear_slot(&fake);
        assert!(matches!(result, Err(MemoryError::NotFound(_))));
    }

    #[test]
    fn test_utilization_math() {
        let wm = WorkingMemory::new(4).expect("new");
        assert!((wm.utilization() - 0.0).abs() < f32::EPSILON);
        wm.focus(item(0.5), SlotPriority::Normal).expect("focus");
        assert!((wm.utilization() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_active_returns_all_items() {
        let wm = WorkingMemory::new(5).expect("new");
        for _ in 0..3 {
            wm.focus(item(0.5), SlotPriority::Normal).expect("focus");
        }
        assert_eq!(wm.active().len(), 3);
    }
}
