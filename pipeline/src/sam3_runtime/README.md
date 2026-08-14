# HonCut SAM 3 continuity tracker

This is a tracking-only service for Phase 8 seam adjudication. The vendored
model source is in `vendor/sam3`; Meta's license is preserved there.

## Setup

```bash
pipeline/scripts/setup_sam3.sh
```

HonCut first checks `pipeline/models/sam3/sam3.pt`, then `SAM3_ASSET_ROOT`, and
finally a sibling `../sam3/权重/sam3.pt`. The current development layout is
therefore detected without copying another 3.2 GB file. To use another shared
directory, set:

```bash
export SAM3_ASSET_ROOT=/absolute/path/to/sam3-assets
```

If no local weights exist, the `facebook/sam3` weights are gated. Accept Meta's
terms on Hugging Face, authenticate with `hf auth login`, then download without
placing the token in the repository:

```bash
mkdir -p pipeline/models/sam3
.venv-sam3/bin/hf download facebook/sam3 sam3.pt \
  --local-dir pipeline/models/sam3
```

The recommended integration lets Phase 8 own the local service lifecycle:

```bash
export HONCUT_SAM3_MODE=managed
```

When Phase 8 reaches internal-seam adjudication it starts SAM 3 on demand,
passes the endpoint directly to the seam adjudicator, and stops the owned
process after the pass. Logs are written to `<output-dir>/logs/SAM3_SIDECAR.log`.
An already healthy `HONCUT_SAM3_URL` is reused and is never stopped by HonCut.

To operate a long-lived service yourself instead:

```bash
pipeline/scripts/start_sam3.sh
export HONCUT_SAM3_MODE=external
export HONCUT_SAM3_URL=http://127.0.0.1:8001
```

Leaving both variables unset keeps SAM 3 off and preserves the previous Phase 8
behavior.

## M4 / 16 GB defaults

- `SAM3_DEVICE=auto` resolves to MPS.
- `SAM3_PRECISION=auto` resolves to stable FP32 on MPS and dynamic INT8/QNNPACK
  on CPU. Whole-model MPS FP16 is not the default because upstream explicitly
  promotes image inputs to FP32 before the visual backbone.
- `HONCUT_SAM3_ANALYSIS_FPS=3` reduces a 24 fps tracking clip to one eighth of
  the frames. A short SAM-bounded template pass then refines the catch-up on the
  original timeline and refuses cuts that would leave under 0.5 seconds.
- Only one tracking session is retained, and `SAM3_MAX_FRAMES=96` prevents an
  accidental long clip from exhausting unified memory.
- `/runtime` reports the selected precision, parameter counts, estimated weight
  memory, model load duration, and process RSS change.

The loaded checkpoint has 860,055,224 parameters. Theoretical weight storage is
3.204 GiB at FP32, 1.602 GiB at FP16, and 1.240 GiB when the 81.7% of parameters
in Linear layers are dynamically quantized to INT8. Activations and framework
overhead are extra.

On 2026-08-14 the real shared `sam3.pt` completed a CPU INT8/QNNPACK smoke test
with a two-frame truck clip: load took 7.13 seconds, the prompt found object `0`
at confidence `0.7365`, and propagation returned stable centroids on both frames.
The managed test sandbox hid MPS, so this proof exercised the CPU fallback; the
native service resolves to MPS/FP32 when MPS is available.

If an MPS operator fails despite `PYTORCH_ENABLE_MPS_FALLBACK=1`, retry with
`SAM3_DEVICE=cpu SAM3_PRECISION=int8_dynamic`. For diagnosis, use
`SAM3_PRECISION=fp32` explicitly. CPU activations remain FP32 for QNNPACK;
`SAM3_CPU_BF16=1` is only appropriate for a non-quantized CPU experiment.
