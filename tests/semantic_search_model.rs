//! Meaning-based recall with a real local model. Downloads BGE-small (about
//! 130 MB) on first run, so it is opt-in:
//! `cargo test --features fastembed --test semantic_search_model -- --ignored`

#![cfg(feature = "fastembed")]

use tokio_agent_memory::retrieval::{retrieve_from_items, MemoryQuery, RetrievalMode};
use tokio_agent_memory::semantic_search::{item_text, FastEmbedder, SemanticIndex};
use tokio_agent_memory::types::{AgentId, MemoryItem};

fn memories() -> Vec<MemoryItem> {
    [
        "User is vegetarian and allergic to peanuts",
        "Booked a window seat on TAP flight 1234 to Lisbon",
        "The quarterly report is due on Friday",
        "User's daughter starts school in September",
        "Prefers replies in short bullet points",
    ]
    .iter()
    .map(|t| {
        let mut m = MemoryItem::new(serde_json::json!(t), vec![], AgentId::new("assistant"), 0.5);
        m.confidence = 1.0;
        m
    })
    .collect()
}

#[tokio::test]
#[ignore]
async fn questions_find_memories_that_share_no_words_with_them() {
    let items = memories();
    let index = SemanticIndex::new(FastEmbedder::try_default().unwrap());
    let cases = [
        ("any dietary restrictions I should know about?", "vegetarian"),
        ("when does the plane leave for Portugal?", "Lisbon"),
        ("how should I format my answer?", "bullet points"),
        ("any deadlines coming up at work?", "quarterly report"),
    ];
    for (question, expected) in cases {
        let hits = index.search(&items, question, 1, 0.0).await.unwrap();
        let top = item_text(&hits[0].item);
        println!("{question:45} -> {top} ({:.2})", hits[0].score);
        assert!(top.contains(expected), "{question}: got {top}");

        // The fuzzy mode cannot answer these: no shared words.
        let fuzzy = retrieve_from_items(
            &items,
            &MemoryQuery { mode: RetrievalMode::Fuzzy { text: question.into(), threshold: 0.3 }, limit: 1, min_confidence: 0.0 },
        )
        .unwrap();
        assert!(fuzzy.first().map_or(true, |r| !item_text(&r.item).contains(expected)));
    }
}
