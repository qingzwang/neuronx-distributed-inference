# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""LoRA support for the FLUX backbone.

Community FLUX LoRAs ship in at least three key conventions -- diffusers/PEFT
(``transformer.transformer_blocks.0.attn.to_q.lora_A.weight``), kohya
(``lora_unet_double_blocks_0_img_attn_proj.lora_down.weight`` plus ``.alpha``),
and XLabs (``double_blocks.0.processor.proj_lora1.down.weight``). Rather than
parse three formats, this module hands the file to
``FluxPipeline.lora_state_dict``, which dispatches to diffusers' own converters,
and works from the diffusers key space that comes back.

From there the mapping onto NxDI is almost free: after dropping the
``transformer.`` prefix, a diffusers key names exactly the module NxDI wrapped,
because the two implementations use the same attribute names
(``attn.to_q``, ``ff.net.0.proj``, ``norm1.linear``, ...).

One module does not line up. diffusers' single-stream block has a single
``proj_out`` whose input is the concatenation ``[attn_out ; mlp_out]``; NxDI
splits it into ``proj_out_attn`` and ``proj_out_mlp`` and sums their outputs. A
LoRA on the diffusers layer therefore has to be split too -- see
:func:`split_single_block_proj_out`, which is exact rather than an
approximation.
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from neuronx_distributed_inference.modules.lora_serving.lora_checkpoint import LoraCheckpoint

logger = logging.getLogger("Neuron")

# Module suffixes a FLUX LoRA can target, as `LoraServingConfig.target_modules`.
# The union of what real adapters touch: a kohya adapter covers all of these, an
# XLabs one only the double-block attention projections. Wrapping a module no
# adapter targets costs an unused slot, so this is the full set rather than a
# guess per adapter.
FLUX_LORA_TARGET_MODULES = [
    # double-stream (MMDiT) blocks
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "norm1.linear",
    "norm1_context.linear",
    # single-stream blocks
    "proj_mlp",
    "norm.linear",
    # NxDI's two halves of diffusers' single-block proj_out
    "proj_out_attn",
    "proj_out_mlp",
]

# diffusers prefixes its transformer keys with this.
_TRANSFORMER_PREFIX = "transformer."

# The one diffusers module NxDI splits in two.
_SINGLE_BLOCK_PROJ_OUT = "proj_out"


def load_flux_lora_state_dict(
    path: str, weight_name: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """Load a FLUX LoRA from disk into the diffusers key space.

    Accepts any format ``FluxPipeline.lora_state_dict`` accepts -- diffusers/PEFT,
    kohya, or XLabs -- and returns keys of the form
    ``transformer.<module path>.lora_{A,B}.weight``.

    Args:
        path: Directory or file holding the adapter.
        weight_name: Specific file inside ``path``, when the directory holds more
            than one adapter.

    Returns:
        The converted state dict.

    Raises:
        ValueError: If the adapter contains nothing for the transformer, which
            means it is either not a FLUX LoRA or targets only the text encoders.
    """
    from diffusers import FluxPipeline

    state_dict = FluxPipeline.lora_state_dict(path, weight_name=weight_name)
    if isinstance(state_dict, tuple):  # newer diffusers also returns metadata
        state_dict = state_dict[0]

    transformer_keys = [k for k in state_dict if k.startswith(_TRANSFORMER_PREFIX)]
    if not transformer_keys:
        raise ValueError(
            f"No transformer weights in the LoRA at {path}. Keys start with "
            f"{sorted({k.split('.')[0] for k in state_dict})}. Either this is not "
            "a FLUX LoRA, or it only adapts the text encoders, which this path "
            "does not cover."
        )
    dropped = len(state_dict) - len(transformer_keys)
    if dropped:
        logger.warning(
            f"Ignoring {dropped} non-transformer tensors in the LoRA at {path} "
            "(text-encoder adapters are not supported for FLUX)."
        )
    return {k: state_dict[k] for k in transformer_keys}


def split_single_block_proj_out(
    lora_a: torch.Tensor, lora_b: torch.Tensor, attn_dim: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a LoRA on diffusers' single-block ``proj_out`` into NxDI's two halves.

    diffusers computes ``proj_out([attn_out ; mlp_out])``; NxDI computes
    ``proj_out_attn(attn_out) + proj_out_mlp(mlp_out)``. The split is exact
    because ``lora_B`` is linear::

        B @ (A @ [x_a ; x_m]) == B @ (A[:, :d] @ x_a) + B @ (A[:, d:] @ x_m)

    so ``lora_A`` is cut along its input dimension while ``lora_B`` is **shared**,
    not cut. The two halves' outputs are summed by the block, reproducing the
    original.

    Args:
        lora_a: ``[rank, attn_dim + mlp_dim]``.
        lora_b: ``[out_dim, rank]``.
        attn_dim: Width of the attention half of the concatenated input.

    Returns:
        ``(attn_half, mlp_half)``, each a ``{"lora_A": ..., "lora_B": ...}`` dict.

    Raises:
        ValueError: If ``lora_a`` is narrower than ``attn_dim``, which means the
            checkpoint does not have the concatenated layout this assumes.
    """
    if lora_a.shape[1] <= attn_dim:
        raise ValueError(
            f"single-block proj_out lora_A has input width {lora_a.shape[1]}, "
            f"which is not wider than the attention half ({attn_dim}); expected "
            "the concatenated [attn ; mlp] layout."
        )
    return (
        {"lora_A": lora_a[:, :attn_dim].contiguous(), "lora_B": lora_b},
        {"lora_A": lora_a[:, attn_dim:].contiguous(), "lora_B": lora_b},
    )


def convert_flux_lora_to_nxdi(
    state_dict: Dict[str, torch.Tensor], attn_dim: int
) -> Dict[str, torch.Tensor]:
    """Rewrite a diffusers-space FLUX LoRA onto NxDI module names.

    Args:
        state_dict: Output of :func:`load_flux_lora_state_dict`.
        attn_dim: The backbone's hidden size, i.e. the width of the attention half
            of the single block's concatenated ``proj_out`` input.

    Returns:
        A state dict whose keys are NxDI's wrapped-module parameter names, e.g.
        ``transformer_blocks.0.attn.to_q.lora_A.weight``, ready for an exact
        lookup by :class:`FluxLoraCheckpoint`.
    """
    out: Dict[str, torch.Tensor] = {}
    proj_out_pairs: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, weight in state_dict.items():
        name = key.removeprefix(_TRANSFORMER_PREFIX)
        module_path, _, tail = name.rpartition(".lora_")
        if not module_path:
            logger.warning(f"Skipping unrecognized LoRA key {key!r}")
            continue
        which = "lora_A" if tail.startswith("A") else "lora_B"

        # The single block's proj_out needs both halves before it can be split,
        # so collect the pair and handle it after the loop.
        if module_path.endswith(f".{_SINGLE_BLOCK_PROJ_OUT}") and module_path.startswith(
            "single_transformer_blocks."
        ):
            proj_out_pairs.setdefault(module_path, {})[which] = weight
            continue

        out[f"{module_path}.{which}.weight"] = weight

    for module_path, pair in proj_out_pairs.items():
        if set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(
                f"{module_path} has only {sorted(pair)}; both lora_A and lora_B "
                "are needed to split it across NxDI's proj_out_attn/proj_out_mlp."
            )
        attn_half, mlp_half = split_single_block_proj_out(
            pair["lora_A"], pair["lora_B"], attn_dim
        )
        base = module_path[: -len(_SINGLE_BLOCK_PROJ_OUT)]
        for suffix, half in (("proj_out_attn", attn_half), ("proj_out_mlp", mlp_half)):
            for which, weight in half.items():
                out[f"{base}{suffix}.{which}.weight"] = weight

    return out


class FluxLoraCheckpoint(LoraCheckpoint):
    """LoraCheckpoint that reads FLUX adapters and matches keys exactly.

    The base class matches a module to its weights heuristically, keying on
    ``layers.`` and otherwise falling back to the bare module type. FLUX names
    contain neither, so every block's ``to_q`` would collapse onto the same key.
    Here the checkpoint is converted to NxDI's own module names up front, which
    makes the lookup exact and removes the ambiguity.
    """

    def __init__(self, config, attn_dim: int):
        """
        Args:
            config: The ``LoraServingConfig``.
            attn_dim: Backbone hidden size, needed to split the single block's
                ``proj_out``.
        """
        super().__init__(config)
        self.attn_dim = attn_dim

    def _load_lora_state_dict_from_path(self, path):
        """Load and convert, in place of the base class's format handling."""
        state_dict = convert_flux_lora_to_nxdi(
            load_flux_lora_state_dict(path), self.attn_dim
        )
        # FLUX LoRAs carry their scaling inside the file (kohya `.alpha`) and
        # diffusers' converters have already folded it into the weights, so there
        # is no separate alpha to apply here.
        return None, state_dict

    def _get_module_checkpoint(self, name, lora_ckpt):
        """Exact-name lookup.

        Args:
            name: NxDI LoRA module parameter name, e.g.
                ``transformer_blocks.0.attn.to_q.lora_A``.
            lora_ckpt: One entry of the loaded checkpoint map.

        Returns:
            The matching tensor, or None when the adapter does not touch this
            module -- which is normal, since adapters target different subsets.
        """
        state_dict = lora_ckpt["state_dict"]
        key = name if name.endswith(".weight") else f"{name}.weight"
        return state_dict.get(key)
