---
type: llm
focus: trace
weight: 1
---

design.md explicitly says not to build |x| (or the sign split) with max(x, -x) because max's jvp splits ties and makes the derivative at x = 0 wrong, and uses where/select on the sign instead
