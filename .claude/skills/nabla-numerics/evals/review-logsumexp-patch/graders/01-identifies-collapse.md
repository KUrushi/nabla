---
type: llm
focus: trace
weight: 1
---

review.md identifies that for a dominant logit (margin above about 16.6 in f32) the patch's logsumexp/cross-entropy rounds the loss and the target gradient to zero while other gradients stay nonzero (Softmax Collapse), so the gradient no longer sums to zero
