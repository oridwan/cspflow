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
