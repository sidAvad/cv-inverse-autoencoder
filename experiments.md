# Experiments

One row per real run. Dry-runs are omitted.

| Run | Date | Model | What changed | Best val loss | Epochs | Verdict |
|---|---|---|---|---|---|---|
| `baseline` | 2026-04-24 | CVSurrogate (decoder) | Initial run — plain BCE on valve signals, MSE on continuous | 0.0754 (ep 10) | 10 | **Kept** — canonical decoder checkpoint |
| `weighted-bce` | 2026-05-05 | CVSurrogate (decoder) | Transition-weighted BCE on valve signals to catch rapid open/close switches | 0.1493 (ep 19) | 20 | **Dropped** — total val loss higher than baseline; valve head loss improved in absolute terms but continuous loss also degraded; weighting scheme needs revisiting |
| `autoenc-baseline` | 2026-05-06 | CVEncoder + frozen CVSurrogate | First encoder run — reconstruction-only loss through frozen baseline decoder | 0.0710 (ep 17) | 20 | **Kept** — canonical encoder checkpoint; slight overfitting after ep 13; Vvs and Pvs are weakest channels (range-norm err >100%) |
