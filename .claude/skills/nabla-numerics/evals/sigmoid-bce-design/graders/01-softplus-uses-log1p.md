---
type: llm
focus: trace
weight: 1
---

design.md computes softplus / log(1 + exp(x)) with log1p (e.g. max(x, 0) + log1p(exp(-|x|))) rather than log(1 + exp(x))
