---
type: llm
focus: trace
weight: 1
---

review.md points out that the sigmoid gradient (sigmoid(x)(1 - sigmoid(x)) or the autodiff of 1/(1+exp(-x))) rounds to 0 or loses accuracy for large |x| because 1 - sigmoid(x) is computed by subtraction, and proposes computing the small factor directly
