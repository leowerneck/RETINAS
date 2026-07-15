# RETINAS Algorithms

## Purpose

Define shared registration vocabulary and route algorithm questions without
claiming backend parity.

## Read First

| Page | Question or use | Primary evidence |
| --- | --- | --- |
| [Registration pipeline](registration-pipeline.md) | How does one frame move from construction/first-frame handling through finalization? | `retinas/pyretinas.py`, `retinas/retinas.py`, native composite functions |
| [Preprocessing and correlation](preprocessing-and-correlation.md) | How do centering, brightness, normal, and shot-noise transforms work? | mode-specific Python/C/CUDA stage functions |
| [Displacement estimation](displacement-estimation.md) | What are axes/signs, full-pixel choice, localized upsampling, and refinement? | estimator/upsampling functions and notebook derivation |
| [Reference update modes](reference-update-modes.md) | How do first-reference, online, accumulation, reset, and continuation differ? | update/accumulation state functions |

Core vocabulary: input image, optional first-frame centering, preprocessing and
brightness, reference representation, correlation product, full-pixel estimate,
localized upsampling, sub-pixel refinement, online reference update, accumulated
image sum, averaged-reference replacement, and finalization. Same term or
symbol across implementations does not establish same state/default/behavior.

## Common Tasks

- For end-to-end control/state flow, read registration pipeline.
- For one stage or update mode, read its leaf above; do not copy pipeline flow.
- For layer boundaries, read [architecture](../architecture.md).
- For backend-specific evidence levels, read
  [implementations](../implementations/index.md).
- For changed paths, use [change impact](../change-impact.md).

## Ownership Boundary

This hub owns shared vocabulary and child routing. Registration pipeline alone
owns cross-layer flow. Stage/mode leaves own focused contracts. Backend
allocation/API details stay with implementation owners; numerical comparison
stays in parity matrix; fixture/oracle
claims stay under testing.

## Evidence Limits And Current Gaps

Current children are based mainly on static source/manifests. No numerical derivation,
native build, CUDA execution, or parity assertion was performed. Shot-noise and
accumulation paths differ in visible evidence and must remain qualified.

Parent: [wiki index](../index.md). Related: [catalog](../catalog.md) and
[testing](../testing/index.md).
