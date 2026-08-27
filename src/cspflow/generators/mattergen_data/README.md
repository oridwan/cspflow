# Vendored MatterGen data files

`gemnet-dT.json` is copied verbatim from `microsoft/mattergen`
(`mattergen/common/gemnet/gemnet-dT.json`), MIT licensed, fetched 2026-08-27.
It holds GemNet-dT's learned activation scaling factors — without it the model
cannot be instantiated at all.

## Two separate reasons it is needed

**The wheel does not ship it.** `pip install mattergen` installs no data files;
`gemnet.py` looks for `{MODELS_PROJECT_ROOT}/common/gemnet/gemnet-dT.json`,
which does not exist in a site-packages install. Same root cause as
`../sampling_conf/README.md`.

**Checkpoints hard-code an absolute path to it.** The campaign checkpoint at
`/projects/mmi/shuo/MatterGen_checkpoints/18-55-08/config.yaml` contains

```yaml
scale_file: /users/stao/apps/mattergen/mattergen/common/gemnet/gemnet-dT.json
```

which is **permission denied** to anyone but its author. Every MatterGen
checkpoint records whatever path the machine that trained it happened to use, so
this is not specific to that one checkpoint — it is what a checkpoint always
does.

`MatterGenEngine` therefore passes
`--config_overrides=[...gemnet.scale_file=<this file>]` on every invocation,
which makes the checkpoint usable by anyone who can read it.
