# tokio-agent-memory

<!-- outcome-block -->
**Agents that remember across hours, not just one context window: episodic, semantic and working memory in-process, with no database or server.**

**Why this over the alternatives:** Vector stores only do similarity search. This adds cause-and-effect links, a fact graph, priority-based working memory and decay, as plain Rust data structures.

**Guarantee:** if it doesn't do what this README says, [open an issue](https://github.com/Mattbusel/tokio-agent-memory/issues) and I'll fix it.
<!-- outcome-block -->


[![crates.io](https://img.shields.io/crates/v/tokio-agent-memory.svg)](https://crates.io/crates/tokio-agent-memory)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Memory for Tokio-based LLM agents: an episodic event log with cause-and-effect links, a semantic fact graph, a fixed-size working memory, confidence decay, retrieval (exact, fuzzy, time range, tags), JSON file persistence and a broadcast bus for sharing memories between agents.

An agent that runs for hours needs more than its context window. This crate gives you the usual memory tiers as plain in-process data structures you can wire into any agent loop, with no database or server required.

## Features

| Module | What it does |
|---|---|
| `episodic` | `EpisodicMemory`: record `Event`s, recall them by time range, follow `caused_by` links with `causal_chain`, `forget` by id |
| `semantic` | `SemanticMemory`: subject, relation, object facts (`Concept`, `Relation`), `query`, multi-hop `related(concept, depth)`, `retract` |
| `working` | `WorkingMemory`: fixed number of slots with `SlotPriority` (Low, Normal, High, Critical); when full, the lowest-priority oldest item is evicted |
| `decay` | `DecayScheduler` with `Exponential` or `Linear` confidence decay and a forget threshold; items with importance above 0.8 are never forgotten |
| `retrieval` | `retrieve_from_items` with `RetrievalMode::Exact` (tag), `Fuzzy` (skim fuzzy matching), `Temporal`, `TagIntersection`, plus a confidence floor and limit |
| `consolidation` | `Consolidator`: moves working-memory items into semantic memory (importance at or above a threshold) or episodic memory (the rest) |
| `persistence` | `MemoryStore` trait with `InMemoryStore` and `FileStore` (a JSON file, written on `flush`) |
| `shared` | `SharedMemoryBus`: Tokio broadcast channel of `MemoryItem`s plus time-limited write leases per key |
| `semantic_search` | `SemanticIndex`: find memories by meaning with embeddings, from a local model (`fastembed` feature) or any OpenAI-compatible API (`openai` feature) |

All types are `Send + Sync` (backed by `DashMap` and `parking_lot`), and the crate denies `unwrap`, `expect` and `panic` through Clippy lints.

## Quick start

```bash
cargo add tokio-agent-memory
cargo add tokio --features full
cargo add serde_json
```

```rust
use std::time::Duration;
use tokio_agent_memory::{
    consolidation::{ConsolidationPolicy, Consolidator},
    episodic::{EpisodicMemory, Event},
    retrieval::{retrieve_from_items, MemoryQuery, RetrievalMode},
    semantic::{Concept, Relation, SemanticMemory},
    shared::SharedMemoryBus,
    types::{AgentId, MemoryItem},
    working::{SlotPriority, WorkingMemory},
    MemoryError,
};

#[tokio::main]
async fn main() -> Result<(), MemoryError> {
    let agent = AgentId::new("planner-1");

    // Episodic: what happened, and what caused what.
    let episodic = EpisodicMemory::new();
    let asked = episodic.record(Event::new("user asked for a trip plan", vec!["user".into()])).await?;
    let searched = episodic
        .record(Event::new("searched flights to Lisbon", vec!["tool".into()]).with_cause(asked))
        .await?;
    println!("chain length: {}", episodic.causal_chain(&searched).await?.len()); // 2

    // Semantic: facts as a small graph.
    let semantic = SemanticMemory::new();
    semantic.assert_fact(Concept::new("Lisbon"), Relation::new("capital_of"), Concept::new("Portugal")).await?;
    println!("{:?}", semantic.query(&Concept::new("Lisbon")).await?);

    // Working memory: a few prioritized slots for the current task.
    let working = WorkingMemory::new(8)?;
    working.focus(MemoryItem::new(serde_json::json!("budget is $800"), vec!["constraint".into()], agent.clone(), 0.9), SlotPriority::High)?;
    working.focus(MemoryItem::new(serde_json::json!("prefers window seats"), vec!["preference".into()], agent.clone(), 0.4), SlotPriority::Normal)?;

    // Fuzzy retrieval over working-memory items.
    let hits = retrieve_from_items(&working.active(), &MemoryQuery {
        mode: RetrievalMode::Fuzzy { text: "budget".into(), threshold: 0.1 },
        limit: 5,
        min_confidence: 0.0,
    })?;
    println!("{} hit(s) for 'budget'", hits.len());

    // Consolidate: important items become facts, the rest become events.
    let report = Consolidator::new(ConsolidationPolicy::default()).run(&working, &episodic, &semantic).await?;
    println!("semantic +{}, episodic +{}", report.promoted_to_semantic, report.promoted_to_episodic);

    // Share with other agents.
    let bus = SharedMemoryBus::new(64);
    let mut rx = bus.subscribe();
    bus.acquire_lease("trip-plan", agent.clone(), Duration::from_secs(30)).await?;
    bus.publish(MemoryItem::new(serde_json::json!("plan v1 ready"), vec![], agent, 0.7)).await?;
    println!("peer received: {}", rx.recv().await.map_err(|_| MemoryError::ChannelClosed)?.content);
    Ok(())
}
```

## Search by meaning

Tag and fuzzy retrieval only find memories that share words with the question. `SemanticIndex` embeds each memory once and ranks memories by cosine similarity to the question, so "any dietary restrictions I should know about?" finds "User is vegetarian and allergic to peanuts".

```toml
tokio-agent-memory = { version = "0.2", features = ["fastembed"] }  # local model, no API key
# or features = ["openai"]  for OpenAI, Ollama, vLLM, LM Studio, text-embeddings-inference
```

```rust,no_run
use tokio_agent_memory::semantic_search::{FastEmbedder, SemanticIndex};
use tokio_agent_memory::types::{AgentId, MemoryItem};

#[tokio::main]
async fn main() -> Result<(), tokio_agent_memory::MemoryError> {
    let memories: Vec<MemoryItem> = [
        "User is vegetarian and allergic to peanuts",
        "Booked a window seat on TAP flight 1234 to Lisbon",
        "The quarterly report is due on Friday",
    ]
    .iter()
    .map(|t| MemoryItem::new(serde_json::json!(t), vec![], AgentId::new("assistant"), 0.5))
    .collect();

    // BGE-small-en-v1.5, downloaded once (about 130 MB) to your cache directory.
    let index = SemanticIndex::new(FastEmbedder::try_default()?);
    for hit in index.search(&memories, "when does the plane leave for Portugal?", 1, 0.0).await? {
        println!("{:.2} {}", hit.score, hit.item.content); // 0.69 "Booked a window seat ... Lisbon"
    }
    Ok(())
}
```

With the `openai` feature, use `OpenAiEmbedder::new("text-embedding-3-small").with_api_key(key)`, or `.with_base_url("http://localhost:11434/v1")` for Ollama. Any other model works through the one-method `Embedder` trait. Each memory is embedded once (in a batch) and cached; only the current items you pass are searched, so forgotten memories do not come back. The `fastembed` feature needs Rust 1.88.

Small local models match paraphrases well (the scores above are from the crate's test against the real model); questions that need a reasoning step ("what can I cook tonight?" for a dietary memory) need a larger embedding model.

## Layout

```text
src/
  types.rs          MemoryId, AgentId, MemoryItem (JSON content, tags, confidence, importance)
  episodic/         EpisodicMemory, Event, CausalChain
  semantic/         SemanticMemory, Concept, Relation, Fact, ConceptGraph
  working/          WorkingMemory, SlotPriority
  decay/            DecayScheduler, DecayPolicy
  retrieval/        retrieve_from_items, MemoryQuery, RetrievalMode
  consolidation/    Consolidator, ConsolidationPolicy
  persistence/      MemoryStore, InMemoryStore, FileStore
  shared/           SharedMemoryBus, WriteLease
  semantic_search.rs  Embedder, SemanticIndex, FastEmbedder, OpenAiEmbedder
tests/memory_integration.rs     cross-module scenarios
tests/semantic_search_model.rs  meaning-based recall against the real model (opt-in)
```

## Status and limitations

Version 0.2. Everything lives in process memory; `FileStore` is the only on-disk option (the semantic index is in memory too). Retrieval is lexical (tags, fuzzy matching) or by meaning (`SemanticIndex`). The `MemoryStore` trait is synchronous. For a similar crate with a trait-based episodic/semantic store design, Ebbinghaus decay and binary snapshots, see [tokio-memory](https://gitlab.com/mattbusel/tokio-memory).

```bash
cargo test --all-features
cargo test --features fastembed --test semantic_search_model -- --ignored   # real model
```

## License

MIT, see [LICENSE](LICENSE).

---

Part of a set of Rust crates for LLM agents, see [rust-crates](https://gitlab.com/mattbusel/rust-crates).
