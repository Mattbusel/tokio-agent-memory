use crate::{error::MemoryError, types::{AgentId, MemoryItem}};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

pub struct WriteLease {
    pub key: String,
    pub holder: AgentId,
    pub expires_at: std::time::Instant,
}

pub struct SharedMemoryBus {
    tx: broadcast::Sender<MemoryItem>,
    leases: Arc<DashMap<String, WriteLease>>,
}

impl SharedMemoryBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            leases: Arc::new(DashMap::new()),
        }
    }

    pub async fn publish(&self, item: MemoryItem) -> Result<(), MemoryError> {
        self.tx.send(item).map(|_| ()).map_err(|_| MemoryError::ChannelClosed)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<MemoryItem> {
        self.tx.subscribe()
    }

    pub async fn acquire_lease(
        &self,
        key: &str,
        agent: AgentId,
        ttl: std::time::Duration,
    ) -> Result<WriteLease, MemoryError> {
        let now = std::time::Instant::now();

        if let Some(existing) = self.leases.get(key) {
            if existing.expires_at > now {
                return Err(MemoryError::LockFailed(key.to_string()));
            }
        }

        let lease = WriteLease {
            key: key.to_string(),
            holder: agent.clone(),
            expires_at: now + ttl,
        };
        self.leases.insert(key.to_string(), WriteLease {
            key: key.to_string(),
            holder: agent,
            expires_at: now + ttl,
        });
        Ok(lease)
    }

    pub async fn release_lease(&self, key: &str, agent: &AgentId) -> Result<(), MemoryError> {
        let held = self.leases.get(key).map(|l| l.holder.clone());
        match held {
            Some(holder) if &holder == agent => {
                self.leases.remove(key);
                Ok(())
            }
            Some(_) => Err(MemoryError::LockFailed(key.to_string())),
            None => Err(MemoryError::NotFound(key.to_string())),
        }
    }

    pub fn has_lease(&self, key: &str, agent: &AgentId) -> bool {
        self.leases
            .get(key)
            .map(|l| &l.holder == agent && l.expires_at > std::time::Instant::now())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    fn make_item() -> MemoryItem {
        MemoryItem::new(serde_json::json!("data"), vec![], AgentId::new("a"), 0.5)
    }

    #[tokio::test]
    async fn test_publish_subscribe() {
        let bus = SharedMemoryBus::new(16);
        let mut rx = bus.subscribe();
        let item = make_item();
        let id = item.id.clone();
        bus.publish(item).await.expect("publish");
        let received = rx.recv().await.expect("recv");
        assert_eq!(received.id, id);
    }

    #[tokio::test]
    async fn test_lease_acquisition() {
        let bus = SharedMemoryBus::new(16);
        let agent = AgentId::new("agent-1");
        let lease = bus
            .acquire_lease("my-key", agent.clone(), std::time::Duration::from_secs(10))
            .await
            .expect("acquire");
        assert_eq!(lease.key, "my-key");
        assert!(bus.has_lease("my-key", &agent));
    }

    #[tokio::test]
    async fn test_lease_conflict_returns_error() {
        let bus = SharedMemoryBus::new(16);
        let a = AgentId::new("a");
        let b = AgentId::new("b");
        bus.acquire_lease("key", a, std::time::Duration::from_secs(60))
            .await
            .expect("first");
        let result = bus
            .acquire_lease("key", b, std::time::Duration::from_secs(60))
            .await;
        assert!(matches!(result, Err(MemoryError::LockFailed(_))));
    }

    #[tokio::test]
    async fn test_expired_lease_can_be_reacquired() {
        let bus = SharedMemoryBus::new(16);
        let a = AgentId::new("a");
        let b = AgentId::new("b");
        // Acquire with zero TTL (immediately expired)
        bus.acquire_lease("key", a, std::time::Duration::from_nanos(1))
            .await
            .expect("first");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        // Should succeed now that lease is expired
        let result = bus
            .acquire_lease("key", b, std::time::Duration::from_secs(60))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_release_lease() {
        let bus = SharedMemoryBus::new(16);
        let agent = AgentId::new("agent");
        bus.acquire_lease("k", agent.clone(), std::time::Duration::from_secs(60))
            .await
            .expect("acquire");
        bus.release_lease("k", &agent).await.expect("release");
        assert!(!bus.has_lease("k", &agent));
    }
}
