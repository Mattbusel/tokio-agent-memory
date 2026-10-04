pub mod error;
pub mod types;
pub mod episodic;
pub mod semantic;
pub mod working;
pub mod consolidation;
pub mod retrieval;
pub mod decay;
pub mod shared;
pub mod persistence;
pub mod semantic_search;

pub use error::MemoryError;
pub use types::{AgentId, MemoryId, MemoryItem};

/// Every Rust example in the README is compiled by `cargo test --doc --all-features`.
#[cfg(all(doctest, feature = "fastembed", feature = "openai"))]
#[doc = include_str!("../README.md")]
mod readme_examples {}
