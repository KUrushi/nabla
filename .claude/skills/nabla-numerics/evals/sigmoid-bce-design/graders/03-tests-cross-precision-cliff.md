---
type: llm
focus: trace
weight: 1
---

design.md's test plan generates large-magnitude logits beyond the f32 precision cliff (about 16.6, or explicitly values like 17 to 80) and, for bf16, beyond about 5.5
