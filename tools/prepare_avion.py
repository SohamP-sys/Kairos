#!/usr/bin/env python3
"""Make AVION's optional FlashAttention imports work on Windows."""

from __future__ import annotations

import argparse
from pathlib import Path


ORIGINAL_IMPORTS = (
    "from flash_attn.modules.mha import MHA as FlashMHA\n"
    "from flash_attn.modules.mlp import Mlp as FlashMlp"
)
OPTIONAL_IMPORTS = """# FlashAttention is only needed when use_flash_attn=True.
try:
    from flash_attn.modules.mha import MHA as FlashMHA
    from flash_attn.modules.mlp import Mlp as FlashMlp
except ModuleNotFoundError as exc:
    if exc.name not in {"flash_attn", "flash_attn.modules"}:
        raise
    FlashMHA = None
    FlashMlp = None"""
MHA_USE = (
    "            self.attn = FlashMHA(d_model, n_head, cross_attn=False, "
    "bias=True, dropout=attn_drop, use_flash_attn=True)"
)
MLP_USE = (
    "            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, "
    "activation=act_layer())"
)


def prepare_avion_source(transformer_path: Path) -> bool:
    source = transformer_path.read_text(encoding="utf-8")
    if OPTIONAL_IMPORTS in source:
        if "if FlashMHA is None:" not in source or "if FlashMlp is None:" not in source:
            raise RuntimeError(
                f"AVION has a partial FlashAttention compatibility patch: {transformer_path}"
            )
        return False
    if source.count(ORIGINAL_IMPORTS) != 1:
        raise RuntimeError(
            f"Unsupported AVION transformer source; expected optional FlashAttention imports: "
            f"{transformer_path}"
        )
    if source.count(MHA_USE) != 1 or source.count(MLP_USE) != 1:
        raise RuntimeError(
            f"Unsupported AVION FlashAttention call sites: {transformer_path}"
        )

    source = source.replace(ORIGINAL_IMPORTS, OPTIONAL_IMPORTS, 1)
    source = source.replace(
        MHA_USE,
        "            if FlashMHA is None:\n"
        "                raise RuntimeError('FlashAttention is required when use_flash_attn=True')\n"
        + MHA_USE,
        1,
    )
    source = source.replace(
        MLP_USE,
        "            if FlashMlp is None:\n"
        "                raise RuntimeError('FlashAttention is required when use_flash_attn=True')\n"
        + MLP_USE,
        1,
    )
    transformer_path.write_text(source, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("avion_root", type=Path)
    args = parser.parse_args(argv)
    transformer = args.avion_root / "avion" / "models" / "transformer.py"
    if not transformer.is_file():
        parser.error(f"AVION transformer source not found: {transformer}")
    try:
        changed = prepare_avion_source(transformer)
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        "Prepared AVION for inference without FlashAttention."
        if changed
        else "AVION FlashAttention compatibility is already prepared."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
