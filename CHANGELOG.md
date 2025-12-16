# Changelog

All notable changes to MemoryPlus will be documented here.

## MemoryPlus v1.1 - 2025-12-14

- **Persistent Graphiti Worker**
  - Added a multiprocessing engine client/worker pair that keeps Graphiti alive between requests.
  - Plugin now supports `engine_mode` with persistent workers, automatic restarts, and subprocess fallback.
- **Queued Ingestion Pipeline**
  - Introduced an internal ingestion queue with configurable capacity, overflow policies, batching, and exponential retries.
  - Graceful shutdown drains pending jobs or logs dropped items during plugin detach.
- **Context Search Cache**
  - Added an LRU-style search cache with TTL and fuzzy matching to avoid redundant Graphiti lookups.
  - Cache invalidates whenever new episodes are written to keep results fresh.
- **NLP Pipeline Improvements**
  - Runner now performs lexicon-based emotion detection, topic tagging, lifecycle pruning, and supports new analysis prompts (ResolveEntities, MemoryGate, CustomPrompt).
  - Insight generation honors custom prompts while maintaining safe defaults.
- **Documentation Refresh**
  - README rewritten with a product overview, setup guide for Kuzu/Neo4j, configuration breakdown, usage tips, and troubleshooting FAQ.
