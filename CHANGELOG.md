# Changelog

## [0.2.0] - 2026-10-04

### Added

- `semantic_search`: `SemanticIndex` finds memories by meaning. It embeds each
  `MemoryItem` once (batched), caches the vectors and ranks by cosine similarity;
  only the items you pass are searched, so forgotten memories never come back.
- `Embedder` trait (with `embed_query` for models that use a query instruction) and
  two implementations: `FastEmbedder` (`fastembed` feature, local ONNX model,
  BGE-small-en-v1.5 with its query instruction by default) and `OpenAiEmbedder`
  (`openai` feature, any OpenAI-compatible `/embeddings` endpoint).
- An opt-in test against the real model: four questions that share no words with
  their memory are all answered correctly, and the fuzzy mode is shown to miss them.
- `MemoryError::Embedding`; `MemoryError` is now `#[non_exhaustive]`.
- Every Rust example in the README is compiled by `cargo test --doc`.

### Changed

- crates.io metadata (description, keywords, categories, repository) is in the
  repository's `Cargo.toml` (0.1.1 was published with it, but it was never committed).
- `rust-version = "1.85"` declared and checked with an MSRV-aware lockfile
  (`fastembed` feature: 1.88).
- `clippy.toml` allows `unwrap`/`expect`/`panic` inside tests, so
  `cargo clippy --all-targets -- -D warnings` passes; library code still denies them.

## [0.1.1]

- First crates.io release.
