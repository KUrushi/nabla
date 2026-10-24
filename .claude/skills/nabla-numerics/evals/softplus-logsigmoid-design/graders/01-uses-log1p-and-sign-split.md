---
type: llm
focus: trace
weight: 1
---

design.md computes softplus as max(x, 0) + log1p(exp(-|x|)) (or an equivalent sign-split form using log1p) and log-sigmoid as -softplus(-x) or an equivalent stable form
