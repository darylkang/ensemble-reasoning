# Vision

## Thesis
Reasoning outputs should be treated as a distribution over decisions/rationales, not a single output.

## Ensemble Reasoning
Ensemble reasoning here means running repeated trials across a configuration distribution and aggregating the induced decision distribution. The ensemble is over trials and configurations (model, prompt framing, temperature, persona, and protocol). Debate and multi-agent interaction are optional and deferred beyond v0.

## Heterogeneity Ladder
- H0: single-shot (one trial per question).
- H1: intra-model sampling (multiple trials via seeds/temperature).
- H2: structured intra-model heterogeneity (personas/roles/framings plus sampling schedules).
- H3: inter-model heterogeneity (multiple providers/models).
- H4: interactive heterogeneity (debate or deliberative protocols).

v0 focuses on H0–H2 only.

## What “Superior” Means
We prioritize calibration and reliability signals, stability of decisions across trials, and visibility into dissenting rationales. Raw accuracy is secondary and never treated as sufficient on its own.

## Non-Goals
- Simulating human populations, juries, or demographics.
- Building a production system or hosted service.
- Creating a commercial product.

## Terminology
Let Q(c) denote an explicit distribution over configurations c (model, persona, prompt, temperature, protocol). The induced decision distribution is written as P_Q(y|x), where y is a normalized decision and x is the question. All reported statistics must reference Q(c).
