# tokio-agent-memory

Tokio-native agent memory system -- episodic, semantic, and working memory with decay scheduling, persistence, and a shared memory bus for multi-agent coordination.

## What's inside

| Module | Description |
|--------|-------------|
| `episodic` | Timestamped event log with causal chain linking, temporal recall, and forget |
| `semantic` | Concept graph with typed assertions, BFS relation traversal, and retraction |
| `working` | Priority-slotted working memory with LRU eviction and capacity control |
| `decay` | `DecayScheduler` with exponential and linear policies; configurable forget thresholds |
| `retrieval` | Multi-strategy retrieval: exact, fuzzy, temporal, tag-based, confidence-filtered |
| `persistence` | `InMemoryStore` and `FileStore` backends behind a `MemoryStore` trait |
| `shared` | `SharedMemoryBus` -- broadcast publish/subscribe with lease-based exclusive access |
| `consolidation` | `Consolidator` pipeline -- moves working memory to episodic or semantic based on importance |

## Features

- **Tokio-native** -- async-first design with `parking_lot` for sync-safe shared state
- **Composable memory tiers** -- mix episodic, semantic, and working memory independently
- **Pluggable persistence** -- swap in Redis, SQLite, or S3 by implementing `MemoryStore`
- **Multi-agent coordination** -- `SharedMemoryBus` with typed leases prevents write conflicts
- **Decay scheduling** -- memories fade naturally; high-importance items survive longer

## Quick start

```rust
use tokio_agent_memory::{
 episodic::EpisodicMemory,
 types::{AgentId, MemoryItem},
};

let memory = EpisodicMemory::new();
let agent = AgentId::new("planner-1");

let item = MemoryItem::new(
 serde_json::json!({"event": "user_login", "user": "alice"}),
 vec!["auth".into(), "session".into()],
 agent.clone(),
 0.8,
);

memory.record(item).unwrap();

let recent = memory.recall_range(
 chrono::Utc::now() - chrono::Duration::hours(1),
 chrono::Utc::now(),
).unwrap();
println!("Recalled {} events", recent.len());
```

## Add to your project

```toml
[dependencies]
tokio-agent-memory = { git = "https://github.com/Mattbusel/tokio-agent-memory" }
```

Or one-liner:

```ash
cargo add --git https://github.com/Mattbusel/tokio-agent-memory
```

## Test coverage

120+ tests across unit, integration, and multi-module pipeline suites.

```bash
cargo test
```

---

> Used inside [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- a production Rust orchestration layer for LLM pipelines. See the full [primitive library collection](https://github.com/Mattbusel/rust-crates).