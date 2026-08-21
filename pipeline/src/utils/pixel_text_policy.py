"""Prevent machine-readable identity metadata from becoming visible pixels."""

from __future__ import annotations

import re


PIXEL_TEXT_METADATA_CONTRACT = (
    "角色合同中的内部编号、序列号、铭文或字母数字标识只作为机器元数据，"
    "不得画进像素；改用无文字的材质、色块、轮廓与妆造锚点保持身份"
)


def strip_pixel_text_identity_markers(description: object) -> str:
    """Keep visual identity while removing instructions to render readable glyphs."""
    result = str(description or "")
    # Character discovery may author wardrobe serials such as ``3C91编号章``.
    # Remove only the visible garment-marking clause; measurements and other
    # numeric body contracts remain untouched because they are not pixel text.
    result = re.sub(
        r"[，,；;]?\s*(?:服装|衣服|衣物|胸口|袖口|衣领|背部|肩部|腰部)"
        r"[^，,。；;]{0,80}?(?:[A-Z0-9][A-Z0-9_-]{1,})"
        r"[^，,。；;]{0,40}?(?:几何)?(?:编号章|编号|字样|铭文|文字|标识|徽标|logo)",
        "",
        result,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", result).strip()
