#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Memory item '{0}' not found")]
    NotFound(String),
    #[error("Working memory at capacity ({0} slots)")]
    CapacityExceeded(usize),
    #[error("Serialization failed: {0}")]
    Serialization(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Lock acquisition failed for key '{0}'")]
    LockFailed(String),
    #[error("Lease expired for key '{0}'")]
    LeaseExpired(String),
    #[error("Concept not found: '{0}'")]
    ConceptNotFound(String),
    #[error("Invalid confidence value {0}: must be 0.0–1.0")]
    InvalidConfidence(f32),
    #[error("Invalid decay rate {0}: must be > 0.0")]
    InvalidDecayRate(f64),
    #[error("Send error: channel closed")]
    ChannelClosed,
    #[error("Working memory capacity must be > 0")]
    ZeroCapacity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_display() {
        let e = MemoryError::NotFound("abc".into());
        assert!(e.to_string().contains("abc"));
    }

    #[test]
    fn test_capacity_exceeded_display() {
        let e = MemoryError::CapacityExceeded(8);
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn test_invalid_confidence_display() {
        let e = MemoryError::InvalidConfidence(1.5);
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn test_invalid_decay_rate_display() {
        let e = MemoryError::InvalidDecayRate(-0.1);
        assert!(e.to_string().contains("-0.1"));
    }

    #[test]
    fn test_lease_expired_display() {
        let e = MemoryError::LeaseExpired("my-key".into());
        assert!(e.to_string().contains("my-key"));
    }

    #[test]
    fn test_channel_closed_display() {
        let e = MemoryError::ChannelClosed;
        assert!(e.to_string().contains("channel closed"));
    }

    #[test]
    fn test_zero_capacity_display() {
        let e = MemoryError::ZeroCapacity;
        assert!(!e.to_string().is_empty());
    }
}
