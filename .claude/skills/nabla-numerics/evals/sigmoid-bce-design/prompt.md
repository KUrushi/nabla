---
tags: [skill-creator]
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Bash]
---

nabla/nn に sigmoid と、ロジットを取る二値交差エントロピー（binary-cross-entropy-with-logits、ラベルは 0〜1 の実数）を足したいです。実装はまだしないで、関数の式（値と勾配がどう計算されるか）と、テストの計画（どんな性質をどんな入力で確かめるか）を設計メモ design.md にまとめてください。
