# cspflow

End-to-end high-throughput crystal structure prediction and first-principles
discovery pipeline.

```
source ──► generate ──► screen ──► reference ──► calibrate ──► filter ──► dft ──► analyze
```

Three entry points converge on one funnel: a chemical space, an explicit list of
compositions, or a list of POSCAR/CIF files (which skips generation entirely).
Cheap MLIP screening runs as a barrier; expensive DFT runs as a throttled stream.

Design, and the reasoning behind every stage:
[`pipeline.md`](/projects/mmi/Ridwan/magnet_s/pipeline.md).

## Install

```bash
pip install -e .
```

## Status

M0 (foundations) in progress.

## Documentation

| Document | What it holds |
|---|---|
| [`pipeline.md`](/projects/mmi/Ridwan/magnet_s/pipeline.md) | The design: every stage, the physics, and the reasoning behind each choice |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Implementation decisions, with the measurement behind each one and the alternatives rejected |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module responsibilities and the dependency rule |
| [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md) | Chronological build record: what was measured, what broke, what changed on disk |

Anything found during implementation that **corrects** the design is written
back into `pipeline.md` and flagged in `DECISIONS.md`, so the two never
disagree.
