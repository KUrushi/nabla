---
type: llm
focus: trace
weight: 1
---

design.md avoids computing the label-1 gradient as sigmoid(x) - 1 by subtraction from 1, and gives a form whose small value is computed directly (e.g. -sigmoid(-x) or exp(-x)/(1+exp(-x)))
