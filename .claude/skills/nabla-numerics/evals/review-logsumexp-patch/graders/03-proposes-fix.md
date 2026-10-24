---
type: llm
focus: trace
weight: 1
---

review.md proposes a concrete fix: log1p of the sum excluding the max with the max slot fixed to 0 (as in nabla.nn's log-softmax), or reusing the existing nabla.nn:log-softmax
