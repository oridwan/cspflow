"""cspflow -- high-throughput crystal structure prediction and first-principles discovery.

The pipeline is a funnel with three entry points (Stage 0 `source`) converging on
one path: generate -> screen (MLIP) -> reference -> calibrate -> filter -> dft ->
analyze.  See pipeline.md for the design and the reasoning behind each stage.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
