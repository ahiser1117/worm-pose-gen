# EXP-SMC-002C — expert visual adjudication

## Purpose

This immutable-style artifact records an explicit expert review of the
30-row EXP-SMC-002B development visual evidence. It does not edit the original
annotation file or either frozen EXP-SMC-001B/002B metrics artifact.

The one-based visual row order is bound to the `per_case` order and SHA-256 of
`experiments/exp_smc_002b_width_relative_fov/metrics.json` in
[`adjudication.json`](adjudication.json).

## Adjudication

- Row 2, `2023-09-19-01-f017959`: the expert judges the existing annotation
  genuinely mistaken. The annotation remains unchanged for provenance.
- Row 22, `2023-10-11-01-f013785`: the expert judges this an expected hard
  failure that downstream SMC should solve, not a required easy-frame anchor.
- The remaining reviewed rows are judged visually good enough to continue
  development.

## Decision and limits

This review supersedes the D-0011 stop for **development continuation only**.
It authorizes development-only construction of the anchor dataset and the
latent, width, renderer, dynamics, and controlled SMC experiments. It does not
erase the prior numeric `NOT_SUPPORTED` decisions, recompute a frozen gate,
establish an annotation noise floor, validate the protected holdout, or
authorize deployment/final-performance claims.

The 2025 holdout remains closed. SMC performance on row 22 and other natural
hard cases remains an open empirical question.
