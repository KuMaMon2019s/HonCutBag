# SAM 3 provenance in HonCut

This directory was copied from `/Users/soda/projects/milimovideo/sam3` on
2026-08-14. The original directory was left untouched. The upstream project is
Meta's `facebookresearch/sam3`; its license is preserved in `LICENSE`.

HonCut does not use the copied `start_sam_server.py`. Its tracking-only,
resource-aware service is maintained in `pipeline/src/sam3_runtime` so upstream
model source remains easy to compare or replace.

HonCut patches `sam3/model_builder.py` to use the standard-library
`importlib.resources` API because current setuptools releases no longer ship
the deprecated runtime `pkg_resources` module.

The runtime dependency list also adds `einops` and `pycocotools`, which are
imported by the inference import graph but were only listed in optional extras.
It adds `socksio` so Hugging Face authentication/downloads work in SOCKS-proxy
environments such as the HonCut development host.
The CPU connected-components fallback uses the already-required headless OpenCV
backend instead of importing optional `scikit-image` when an object is detected.

The CPU tracker defaults to FP32 activations because QNNPACK dynamic INT8 does
not accept BF16 autocast output. Non-quantized experiments can opt back in with
`SAM3_CPU_BF16=1`.
The gathered tracker backbone feature pyramid follows the same dtype policy so
FP32 Conv layers do not receive explicitly recast BF16 features.
