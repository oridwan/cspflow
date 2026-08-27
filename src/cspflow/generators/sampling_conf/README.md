# Vendored MatterGen sampling configs

These two files are copied verbatim from
`microsoft/mattergen` (`sampling_conf/csp.yaml`, `sampling_conf/default.yaml`),
MIT licensed, fetched 2026-08-27.

## Why they are here

`mattergen` 1.0.3 as published on PyPI **ships no Hydra configs at all** — the
sdist contains not one `.yaml` file. But `mattergen/common/utils/globals.py`
points its sampling-config path at

```python
DEFAULT_SAMPLING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "sampling_conf"
```

which, for a normal install, is `<site-packages>/sampling_conf` — a directory
that does not exist. So `mattergen-generate` cannot run at all from a
`pip install mattergen`:

```
hydra.errors.MissingConfigException: Primary config directory not found.
Check that the config directory '.../site-packages/sampling_conf' exists and readable
```

The path resolves correctly only for an **editable install from a git clone**,
where `parents[3]` is the repository root.

Found by submitting the first live generation job: it reached the GPU, imported
cleanly, invoked `mattergen-generate`, and died here in 7.7 seconds.

## How they are used

`MatterGenEngine` prefers the installed package's own directory and falls back
to this copy only when that directory is missing, passing
`--sampling_config_path`. `csp doctor` reports which one is in use, because
"cspflow is supplying MatterGen's own configuration" is not something that
should be silent.

If you later install mattergen editable from a clone, this copy stops being
used automatically.
