//! Recall by meaning: embed memory items and find the ones closest to a
//! question, even when they share no words with it.
//!
//! [`SemanticIndex`] embeds each [`MemoryItem`] once (batched), keeps the
//! vectors, and ranks items by cosine similarity to a query. It works with
//! any [`Embedder`]; two come with the crate:
//!
//! - [`FastEmbedder`] (`fastembed` feature): a local ONNX model, no API key
//!   or network after the first download.
//! - [`OpenAiEmbedder`] (`openai` feature): any OpenAI-compatible
//!   `/embeddings` endpoint (OpenAI, Ollama, vLLM, LM Studio, text-embeddings-inference).
//!
//! ```no_run
//! # #[cfg(feature = "fastembed")]
//! # async fn demo(items: Vec<tokio_agent_memory::MemoryItem>) -> Result<(), tokio_agent_memory::MemoryError> {
//! use tokio_agent_memory::semantic_search::{FastEmbedder, SemanticIndex};
//!
//! let index = SemanticIndex::new(FastEmbedder::try_default()?);
//! // "User is vegetarian" comes back, though no word overlaps.
//! for hit in index.search(&items, "what food does the user eat?", 3, 0.0).await? {
//!     println!("{:.2} {}", hit.score, hit.item.content);
//! }
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use dashmap::DashMap;

use crate::error::MemoryError;
use crate::retrieval::RetrievalResult;
use crate::types::{MemoryId, MemoryItem};

/// Turns text into vectors. Implement it for any embedding model.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// One vector per input text, in order.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError>;

    /// The vector for a search query. Defaults to embedding the text as is;
    /// models trained for asymmetric retrieval (BGE, E5, nomic) override it
    /// to add their query instruction.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, MemoryError> {
        self.embed_batch(&[query.to_owned()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("no vector for the query".into()))
    }
}

/// The text of a memory item that gets embedded: string content as is,
/// anything else as JSON.
pub fn item_text(item: &MemoryItem) -> String {
    match &item.content {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Embedding vectors for memory items, searchable by meaning.
pub struct SemanticIndex<E: Embedder> {
    embedder: E,
    vectors: DashMap<MemoryId, Vec<f32>>,
}

impl<E: Embedder> std::fmt::Debug for SemanticIndex<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticIndex").field("indexed", &self.vectors.len()).finish()
    }
}

impl<E: Embedder> SemanticIndex<E> {
    /// An empty index using `embedder`.
    pub fn new(embedder: E) -> Self {
        Self { embedder, vectors: DashMap::new() }
    }

    /// Embed every item not indexed yet, in one batch. Returns how many
    /// were added.
    pub async fn index(&self, items: &[MemoryItem]) -> Result<usize, MemoryError> {
        let missing: Vec<&MemoryItem> = items.iter().filter(|i| !self.vectors.contains_key(&i.id)).collect();
        if missing.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = missing.iter().map(|i| item_text(i)).collect();
        let vectors = self.embedder.embed_batch(&texts).await?;
        if vectors.len() != missing.len() {
            return Err(MemoryError::Embedding(format!(
                "embedder returned {} vectors for {} texts",
                vectors.len(),
                missing.len()
            )));
        }
        for (item, v) in missing.iter().zip(vectors) {
            self.vectors.insert(item.id.clone(), v);
        }
        Ok(texts.len())
    }

    /// The `limit` items most similar in meaning to `query`, best first.
    ///
    /// Items not indexed yet are embedded first. Only `items` are searched
    /// (pass the current contents of your memory, so forgotten items never
    /// come back); items below `min_confidence` are skipped. Scores are
    /// cosine similarities.
    pub async fn search(
        &self,
        items: &[MemoryItem],
        query: &str,
        limit: usize,
        min_confidence: f32,
    ) -> Result<Vec<RetrievalResult>, MemoryError> {
        self.index(items).await?;
        let q = self.embedder.embed_query(query).await?;
        let mut hits: Vec<RetrievalResult> = items
            .iter()
            .filter(|i| i.confidence >= min_confidence)
            .filter_map(|i| {
                let v = self.vectors.get(&i.id)?;
                Some(RetrievalResult { item: i.clone(), score: cosine(&q, &v) })
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit);
        Ok(hits)
    }

    /// Drop the stored vector for `id` (call when the item is forgotten or
    /// its content changes).
    pub fn forget(&self, id: &MemoryId) -> bool {
        self.vectors.remove(id).is_some()
    }

    /// Number of indexed items.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

// ── fastembed ────────────────────────────────────────────────────────────────

/// A local embedding model run with [fastembed](https://crates.io/crates/fastembed)
/// (ONNX Runtime). `fastembed` feature; needs Rust 1.88.
#[cfg(feature = "fastembed")]
#[derive(Clone)]
pub struct FastEmbedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    query_prefix: String,
}

#[cfg(feature = "fastembed")]
impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder").finish_non_exhaustive()
    }
}

#[cfg(feature = "fastembed")]
impl FastEmbedder {
    /// The instruction BGE v1.5 models expect in front of search queries.
    pub const BGE_QUERY_PREFIX: &'static str = "Represent this sentence for searching relevant passages: ";

    /// `BGE-small-en-v1.5` (384 dimensions), downloaded once (about 130 MB)
    /// into [`default_cache_dir`](Self::default_cache_dir), with its query
    /// instruction ([`BGE_QUERY_PREFIX`](Self::BGE_QUERY_PREFIX)) applied to
    /// queries.
    pub fn try_default() -> Result<Self, MemoryError> {
        Ok(Self::try_new(
            fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_cache_dir(Self::default_cache_dir()),
        )?
        .with_query_prefix(Self::BGE_QUERY_PREFIX))
    }

    /// Any fastembed text model and options (no query prefix; add the one
    /// your model documents with [`with_query_prefix`](Self::with_query_prefix)).
    pub fn try_new(options: fastembed::TextInitOptions) -> Result<Self, MemoryError> {
        let model = fastembed::TextEmbedding::try_new(options)
            .map_err(|e| MemoryError::Embedding(format!("fastembed: {e}")))?;
        Ok(Self { model: std::sync::Arc::new(std::sync::Mutex::new(model)), query_prefix: String::new() })
    }

    /// Text put in front of every search query (not in front of memories).
    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = prefix.into();
        self
    }

    /// `FASTEMBED_CACHE_DIR` when set, otherwise `tokio-agent-memory/fastembed`
    /// in the user's cache directory (fastembed's own default is the current
    /// directory).
    pub fn default_cache_dir() -> std::path::PathBuf {
        if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR") {
            return dir.into();
        }
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        };
        base.unwrap_or_else(std::env::temp_dir).join("tokio-agent-memory").join("fastembed")
    }
}

#[cfg(feature = "fastembed")]
#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let model = std::sync::Arc::clone(&self.model);
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            let mut model = model.lock().map_err(|_| MemoryError::Embedding("model lock poisoned".into()))?;
            model.embed(texts, None).map_err(|e| MemoryError::Embedding(format!("fastembed: {e}")))
        })
        .await
        .map_err(|e| MemoryError::Embedding(format!("embedding task failed: {e}")))?
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, MemoryError> {
        self.embed_batch(&[format!("{}{query}", self.query_prefix)])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("no vector for the query".into()))
    }
}

// ── OpenAI-compatible HTTP ───────────────────────────────────────────────────

/// Embeddings from an OpenAI-compatible `/embeddings` endpoint. `openai` feature.
///
/// ```no_run
/// # #[cfg(feature = "openai")] {
/// use tokio_agent_memory::semantic_search::OpenAiEmbedder;
///
/// let openai = OpenAiEmbedder::new("text-embedding-3-small")
///     .with_api_key(std::env::var("OPENAI_API_KEY").unwrap());
/// let ollama = OpenAiEmbedder::new("nomic-embed-text").with_base_url("http://localhost:11434/v1");
/// # }
/// ```
#[cfg(feature = "openai")]
#[derive(Clone)]
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

#[cfg(feature = "openai")]
impl std::fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "openai")]
impl OpenAiEmbedder {
    /// `model` on OpenAI's API (set a key with [`with_api_key`](Self::with_api_key)).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            api_key: None,
        }
    }

    /// Point at another OpenAI-compatible server (`.../v1`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    /// Send `Authorization: Bearer <key>`.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let mut req = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .json(&serde_json::json!({ "model": self.model, "input": texts }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| MemoryError::Embedding(e.to_string()))?;
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.map_err(|e| MemoryError::Embedding(e.to_string()))?;
        if !status.is_success() {
            return Err(MemoryError::Embedding(format!("embeddings API {status}: {body}")));
        }
        parse_openai_embeddings(&body, texts.len())
    }
}

/// Read `data[*].embedding` in `index` order.
#[cfg(any(feature = "openai", test))]
fn parse_openai_embeddings(body: &serde_json::Value, expected: usize) -> Result<Vec<Vec<f32>>, MemoryError> {
    let data = body["data"]
        .as_array()
        .ok_or_else(|| MemoryError::Embedding(format!("no data array in {body}")))?;
    let mut out: Vec<(usize, Vec<f32>)> = data
        .iter()
        .enumerate()
        .map(|(pos, d)| {
            let index = d["index"].as_u64().map(|i| i as usize).unwrap_or(pos);
            let v = d["embedding"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                .unwrap_or_default();
            (index, v)
        })
        .collect();
    out.sort_by_key(|(i, _)| *i);
    if out.len() != expected {
        return Err(MemoryError::Embedding(format!("{} vectors for {expected} texts", out.len())));
    }
    Ok(out.into_iter().map(|(_, v)| v).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Maps words to fixed axes, so "food" and "vegetarian" can be made close.
    struct Toy {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Embedder for Toy {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| {
                    let t = t.to_lowercase();
                    let has = |ws: &[&str]| ws.iter().any(|w| t.contains(w)) as u8 as f32;
                    vec![has(&["food", "eat", "vegetarian", "meal"]), has(&["flight", "lisbon", "seat"]), 0.1]
                })
                .collect())
        }
    }

    fn item(text: &str) -> MemoryItem {
        let mut i = MemoryItem::new(serde_json::json!(text), vec![], AgentId::new("a"), 0.5);
        i.confidence = 1.0;
        i
    }

    #[tokio::test]
    async fn finds_by_meaning_and_embeds_each_item_once() {
        let index = SemanticIndex::new(Toy { calls: AtomicUsize::new(0) });
        let items = vec![item("User is vegetarian"), item("Window seat on the Lisbon flight")];
        let hits = index.search(&items, "what does the user eat?", 1, 0.0).await.unwrap();
        assert_eq!(item_text(&hits[0].item), "User is vegetarian");
        index.search(&items, "flight?", 1, 0.0).await.unwrap();
        // 1 batch for the items + 1 per query; items were not re-embedded.
        assert_eq!(index.embedder.calls.load(Ordering::SeqCst), 3);
        assert_eq!(index.len(), 2);
    }

    #[tokio::test]
    async fn only_current_items_and_confident_items_are_returned() {
        let index = SemanticIndex::new(Toy { calls: AtomicUsize::new(0) });
        let mut items = vec![item("User is vegetarian"), item("Meal plan: vegan")];
        index.index(&items).await.unwrap();
        items[1].confidence = 0.1;
        let hits = index.search(&items, "food", 5, 0.5).await.unwrap();
        assert_eq!(hits.len(), 1);
        let hits = index.search(&items[..0], "food", 5, 0.0).await.unwrap();
        assert!(hits.is_empty(), "forgotten items are not resurrected from the index");
        assert!(index.forget(&items[0].id));
    }

    #[test]
    fn openai_response_is_read_in_index_order() {
        let body = serde_json::json!({ "data": [
            { "index": 1, "embedding": [0.0, 1.0] },
            { "index": 0, "embedding": [1.0, 0.0] }
        ]});
        let v = parse_openai_embeddings(&body, 2).unwrap();
        assert_eq!(v[0], vec![1.0, 0.0]);
        assert!(parse_openai_embeddings(&body, 3).is_err());
    }
}
