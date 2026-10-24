---
type: llm
focus: trace
weight: 1
---

review.md points out that the naive sigmoid breaks for large-magnitude inputs (e.g. the gradient becomes NaN from 0 * inf or the value/gradient rounds to 0 for very negative x in f32), shows or derives where, and proposes a sign-split form that computes the small quantity directly
