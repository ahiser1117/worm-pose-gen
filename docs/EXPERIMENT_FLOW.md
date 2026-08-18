# Experiment Flow

The project-level decision flow will be updated as evidence accumulates. The
current branch is deliberately short: environment validation precedes data
forensics, and no model claim exists before the split and evaluation gates are
frozen.

```text
Reproducible CUDA + uv bootstrap [complete]
                    |
                    v
Bounded source audit + leakage-safe split [complete: 4/12 readable]
                    |
                    v
Classical proxy-label baseline + synthetic crop benchmark
                    |
                    v
Representation -> temporal context -> optional refinement/uncertainty
                    |
                    v
Frozen selection -> untouched test -> packaged final system
```

The final version will reference the canonical project-level decision-path
figure and decisive visuals for every supported, rejected, revised, or
inconclusive major hypothesis.

The audit forced one important revision: cross-project/background validation
is unavailable because all readable inputs are in the starvation family.
Development therefore uses whole-session separation and reserves the distinct
2025 Hamamatsu-condition session as a one-time, explicitly limited final test.
