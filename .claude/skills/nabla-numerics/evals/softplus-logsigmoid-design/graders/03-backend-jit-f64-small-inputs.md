---
type: llm
focus: trace
weight: 1
---

design.md's test plan checks jitted results on both IREE and PJRT including f64 with small-magnitude inputs, because a backend can lower log1p/expm1 or fold expressions back into naive forms
