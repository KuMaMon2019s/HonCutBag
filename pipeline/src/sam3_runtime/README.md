# HonCut SAM 3 continuity tracker

This is a tracking-only service for Phase 8 seam adjudication. The vendored
model source is in `vendor/sam3`; Meta's license is preserved there.

## Setup

```bash
pipeline/scripts/setup_sam3.sh
```

The `facebook/sam3` weights are gated. Accept Meta's terms on Hugging Face,
authenticate with `hf auth login`, then download without placing the token in
the repository:

```bash
mkdir -p pipeline/models/sam3
.venv-sam3/bin/hf download facebook/sam3 sam3.pt \
  --local-dir pipeline/models/sam3
```

Start the local-only service and enable Phase 8:

```bash
pipeline/scripts/start_sam3.sh
export HONCUT_SAM3_URL=http://127.0.0.1:8001
```

## M4 / 16 GB defaults

- `SAM3_DEVICE=auto` resolves to MPS.
- `SAM3_PRECISION=auto` resolves to FP16 on MPS and dynamic INT8/QNNPACK on CPU.
- `HONCUT_SAM3_ANALYSIS_FPS=6` reduces a 24 fps tracking clip to one quarter of
  the frames; trim decisions are converted back to timeline frames.
- Only one tracking session is retained, and `SAM3_MAX_FRAMES=96` prevents an
  accidental long clip from exhausting unified memory.
- `/runtime` reports the selected precision, parameter counts, estimated weight
  memory, model load duration, and process RSS change.

The copied architecture has 860,055,224 parameters. Measured without checkpoint
weights on this M4 host, theoretical weight storage is 3.204 GiB at FP32,
1.602 GiB at FP16, and 1.317 GiB when the 78.5% of parameters in Linear layers
are dynamically quantized to INT8. Activations and framework overhead are extra.

If an MPS operator fails despite `PYTORCH_ENABLE_MPS_FALLBACK=1`, retry with
`SAM3_DEVICE=cpu SAM3_PRECISION=int8_dynamic`. For diagnosis, use
`SAM3_PRECISION=fp32` explicitly.
