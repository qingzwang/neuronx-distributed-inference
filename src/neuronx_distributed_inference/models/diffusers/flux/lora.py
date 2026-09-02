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
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, mappings

from neuronx_distributed_inference.modules.lora_serving.config import LoraServingConfig
from neuronx_distributed_inference.modules.lora_serving.lora_checkpoint import LoraCheckpoint
from neuronx_distributed_inference.modules.lora_serving.lora_layer import BaseMultiLora
from neuronx_distributed_inference.modules.lora_serving.lora_model import LoraModel, LoraWeightManager
from neuronx_distributed_inference.modules.lora_serving.lora_module import (
    MultiLoraModule,
    MultiLoraModuleColumnParallelLinear,
    MultiLoraModuleRowParallelLinear,
)

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
        """Load and convert, in place of the base class's format handling.

        Raises:
            ValueError: If the adapter's rank exceeds the compiled slot width. The
                slots are sized at ``max_lora_rank``, so a wider adapter cannot be
                loaded and would otherwise be silently truncated.
        """
        state_dict = convert_flux_lora_to_nxdi(
            load_flux_lora_state_dict(path), self.attn_dim
        )
        rank = flux_lora_rank(state_dict)
        max_rank = self.lora_config.max_lora_rank
        if rank > max_rank:
            raise ValueError(
                f"The LoRA at {path} has rank {rank}, above the max_lora_rank "
                f"{max_rank} the slots were built for. Rebuild with "
                f"max_lora_rank={rank} or higher."
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


def flux_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    """The rank of a converted FLUX LoRA, i.e. the widest ``lora_A`` it contains.

    Args:
        state_dict: Output of :func:`convert_flux_lora_to_nxdi`.

    Returns:
        The largest rank across modules. Adapters are usually uniform, but the max
        is what the slots have to accommodate.
    """
    ranks = [w.shape[0] for k, w in state_dict.items() if ".lora_A." in k]
    if not ranks:
        raise ValueError("No lora_A tensors in the converted state dict.")
    return max(ranks)


# The single block builds these two RowParallelLinears with reduce_output=False so
# that their two all-reduces collapse into one, done by the block on their sum
# (modeling_flux.py, NeuronFluxSingleTransformerBlock). proj_out_attn additionally
# uses skip_bias_add=True and so returns a (output, bias) tuple. Both break the
# stock LoRA module, and both need FluxLoraModuleRowParallelLinearNoReduce.
NO_REDUCE_MODULES = ("proj_out_attn", "proj_out_mlp")


def _with_sequence_dim(forward, x, *args, **kwargs):
    """Run ``forward`` on ``x``, giving it a sequence dimension if it lacks one.

    ``BaseMultiLora._einsum_forward`` contracts ``"bij,bjk->bik"``, so it needs a
    rank-3 input. Most FLUX activations are ``[batch, seq, dim]`` and fine, but the
    AdaLayerNorm projections (``norm1.linear``, ``norm1_context.linear``,
    ``norm.linear``) act on the timestep embedding, which is ``[batch, dim]``.
    Without this, HLO generation dies in XLA with
    ``Check failed: dim_to_delete < state.dimensions.size()``.

    A length-1 sequence axis is added and removed again, which changes nothing for
    the base layer either -- a linear broadcasts over leading dimensions.
    """
    if x.dim() != 2:
        return forward(x, *args, **kwargs)

    result = forward(x.unsqueeze(1), *args, **kwargs)
    if isinstance(result, tuple):
        output, bias = result
        return output.squeeze(1), bias
    return result.squeeze(1)


class _OptionalAdapterIds:
    """Makes ``adapter_ids`` optional on a LoRA module's forward.

    The stock modules take it positionally, but the FLUX backbone calls its linear
    layers as ``self.to_q(x)`` -- there are dozens of such call sites across the
    attention, block and normalization classes, and threading an argument through
    all of them would be a large, mechanical edit of code that has nothing to do
    with LoRA.

    Passing None costs nothing here. ``adapter_ids`` reaches
    ``BaseMultiLora.get_weight``, which with ``is_context_encoding=True`` and
    continuous batching off returns ``updated_weight`` without consulting it. The
    slot has already been chosen, once per forward, by
    ``NeuronFluxTransformer2DModel._select_lora_adapters``. Which is also why
    :func:`build_flux_lora_config` pins ``is_context_encoding=True``.
    """

    def forward(self, x, adapter_ids=None, *args, **kwargs):
        return _with_sequence_dim(
            lambda t: super(_OptionalAdapterIds, self).forward(
                t, adapter_ids, *args, **kwargs
            ),
            x,
        )


class FluxLoraModuleColumnParallelLinear(
    _OptionalAdapterIds, MultiLoraModuleColumnParallelLinear
):
    """Column-parallel LoRA that follows the base layer's ``gather_output``.

    The stock module's ``lora_B`` is always a non-gathering
    ``MultiLoraColumnParallelLinear``, which is right for the LLM layers it was
    written for -- those keep their output sharded. FLUX uses both settings: the
    attention and MLP projections do not gather, but the AdaLayerNorm projections
    (``norm1.linear``, ``norm1_context.linear``, ``norm.linear``) and the
    embedders do. Against a gathering base, the un-gathered LoRA output is a
    per-rank slice and the addition fails outright::

        Shapes are not compatible for broadcasting: bf16[1,1,18432] vs bf16[1,1,4608]

    So when the base gathers, gather the LoRA contribution too.
    """

    def create_lora(self):
        super().create_lora()
        if getattr(self.get_base_layer(), "gather_output", False):
            self.lora_B.forward = self._gathered_forward(self.lora_B)

    @staticmethod
    def _gathered_forward(lora_b):
        """Return lora_B's forward with an all-gather appended."""
        inner = lora_b.forward

        def forward(x, adapter_ids, is_context_encoding):
            return mappings.gather_from_tensor_model_parallel_region(
                inner(x, adapter_ids, is_context_encoding)
            )

        return forward


class FluxLoraModuleRowParallelLinear(
    _OptionalAdapterIds, MultiLoraModuleRowParallelLinear
):
    """Row-parallel LoRA whose forward does not require ``adapter_ids``."""


class FluxLoraModuleRowParallelLinearNoReduce(MultiLoraModuleRowParallelLinear):
    """LoRA for a RowParallelLinear whose output is left un-reduced.

    Two things differ from the stock row-parallel LoRA module.

    **No internal all-reduce.** The stock ``lora_A`` reduces its own output, which
    would be wrong here: the base layer's output is a per-rank partial, and the
    block all-reduces ``out_attn + out_mlp`` once afterwards. Reducing inside the
    LoRA would put an already-summed contribution through that second reduce and
    scale it by the TP degree. Deferring instead is exact, because ``lora_B`` is
    replicated across ranks and so commutes with the sum::

        AllReduce_r( B @ (A_r @ x_r) ) == B @ AllReduce_r( A_r @ x_r ) == B @ (A @ x)

    That identity is why :func:`build_flux_lora_config` refuses
    ``lora_shard_linear_layer=True`` -- a column-sharded ``lora_B`` differs per
    rank and cannot be moved outside the reduce.

    **Tuple-returning base layer.** ``proj_out_attn`` has ``skip_bias_add=True``
    and hands back ``(output, bias)``. The bias is added by the block after the
    reduce, so it is passed straight through and the LoRA contribution goes onto
    the output half only.
    """

    def create_lora(self):
        super().create_lora()
        # lora_A is a MultiLoraRowParallelLinear, whose forward ends in an
        # all-reduce. Strip it: the block's reduce covers this contribution.
        self.lora_A.forward = self._lora_a_forward_without_reduce(self.lora_A)
        # The single block reads `self.proj_out_attn.tensor_parallel_group` to
        # perform that reduce. Wrapping put this module in the way, so forward the
        # attribute to the layer that actually has it.
        self.tensor_parallel_group = self.get_base_layer().tensor_parallel_group

    @staticmethod
    def _lora_a_forward_without_reduce(lora_a):
        """Return lora_A's forward with the trailing all-reduce removed."""

        def forward(x, adapter_ids, is_context_encoding):
            weights = lora_a.get_weight(adapter_ids, is_context_encoding)
            return lora_a._einsum_forward(x, weights)

        return forward

    def forward(self, x, adapter_ids=None, *args, **kwargs):
        return _with_sequence_dim(
            lambda t: self._forward_no_reduce(t, adapter_ids, *args, **kwargs), x
        )

    def _forward_no_reduce(self, x, adapter_ids, *args, **kwargs):
        base_output = self.get_base_layer()(x, *args, **kwargs)

        bias = None
        if isinstance(base_output, tuple):
            base_output, bias = base_output

        delta = self.lora_B(
            self.lora_A(x, adapter_ids, self.is_context_encoding),
            adapter_ids,
            self.is_context_encoding,
        )
        result = base_output + delta.to(base_output.dtype)

        return (result, bias) if bias is not None else result


class _FluxLoraModel(LoraModel):
    """LoraModel that routes the single block's proj_out halves to the no-reduce module."""

    def _create_new_module(self, parent, target, current_key):
        if isinstance(target, RowParallelLinear):
            if current_key.endswith(NO_REDUCE_MODULES):
                logger.debug(f"Using the no-reduce LoRA module for {current_key}")
                return FluxLoraModuleRowParallelLinearNoReduce(target, self.lora_config)
            return FluxLoraModuleRowParallelLinear(target, self.lora_config)
        if isinstance(target, ColumnParallelLinear):
            return FluxLoraModuleColumnParallelLinear(target, self.lora_config)
        return super()._create_new_module(parent, target, current_key)


def _align_lora_dtype(model, torch_dtype):
    """Make the LoRA layers report the dtype their weights will actually have.

    A LoRA layer records its dtype from the base layer it wraps. FLUX builds its
    linears in float32 and casts the whole backbone afterwards
    (``ModelWrapperFluxBackbone.get_model_instance``), so the recorded dtype is
    float32 while the parameters end up ``torch_dtype``. Nothing notices until a
    dynamic swap, which allocates the host-side buffer from the recorded dtype and
    then copies it into a device tensor of the real one:

        RuntimeError: Expected self.dtype() == dst.dtype() to be true

    Overwriting it here keeps the two allocations in agreement. The attribute is
    only read when allocating weights, never in the forward pass.
    """
    for module in model.modules():
        if isinstance(module, BaseMultiLora):
            module.dtype = torch_dtype
        if isinstance(module, MultiLoraModule):
            module.lora_dtype = torch_dtype


def wrap_flux_backbone_with_lora(model, lora_config):
    """Inject LoRA adapters into a FLUX backbone.

    The stock ``wrap_model_with_lora`` is not used because the FLUX backbone needs
    the no-reduce module for two of its layers, and because
    ``NeuronFluxTransformer2DModel`` derives from ``torch.nn.Module`` rather than
    ``NeuronBaseModel``, so it never reaches the hook that would have called it.

    Args:
        model: The ``NeuronFluxTransformer2DModel``.
        lora_config: A ``LoraServingConfig``; see :func:`build_flux_lora_config`.

    Returns:
        The ``LoraWeightManager``, or None when ``lora_config`` is None.

    Raises:
        ValueError: If ``lora_shard_linear_layer`` is set, which would break the
            no-reduce module's correctness argument.
    """
    if lora_config is None:
        return None
    if lora_config.lora_shard_linear_layer:
        raise ValueError(
            "FLUX LoRA needs lora_shard_linear_layer=False. The single block's "
            "proj_out halves defer their all-reduce to the block, which is only "
            "exact when lora_B is replicated across ranks; a column-sharded "
            "lora_B differs per rank and cannot be moved outside the reduce."
        )
    _FluxLoraModel(model, lora_config)
    _align_lora_dtype(model, model.config.neuron_config.torch_dtype)
    # neuronx_distributed's preprocess_checkpoint tests for this attribute and, if
    # present, calls model.update_weights_for_lora(checkpoint) during sharding.
    model.lora_wrapped_model = True

    manager = LoraWeightManager(lora_config, model)
    # The manager builds a plain LoraCheckpoint, which cannot read FLUX adapters or
    # match FLUX module names. Swap in the FLUX one.
    attn_dim = model.config.num_attention_heads * model.config.attention_head_dim
    manager.lora_checkpoint = FluxLoraCheckpoint(lora_config, attn_dim)
    return manager


class FluxLoraServingConfig(LoraServingConfig):
    """LoraServingConfig that does not go looking for an ``adapter_config.json``.

    The base class introspects every declared adapter to derive ``target_modules``
    and ``max_lora_rank``, and raises if the folder has no ``adapter_config.json``.
    Real FLUX LoRAs are bare ``.safetensors`` in kohya or XLabs layout and carry no
    such file, so that introspection cannot work here.

    Instead the FLUX module set is used verbatim and ``max_lora_rank`` is taken as
    given. Reading the true rank would mean loading every adapter -- 600 MB each --
    during config construction, so it is validated when the weights are actually
    loaded, by :meth:`FluxLoraCheckpoint._load_lora_state_dict_from_path`.
    """

    def get_lora_config_from_ckpt_paths(self):
        return {
            "target_modules": list(FLUX_LORA_TARGET_MODULES),
            "max_lora_rank": self.max_lora_rank,
        }


def build_flux_lora_config(
    max_loras: int = 1,
    max_lora_rank: int = 64,
    max_cpu_loras: int = 8,
    lora_ckpt_paths=None,
    dynamic_multi_lora: bool = False,
    enable_base_model_only: bool = True,
    **kwargs,
):
    """A ``LoraServingConfig`` with the settings the FLUX backbone requires.

    Args:
        max_loras: Adapters resident on device.
        max_lora_rank: Largest rank to reserve slots for. Real FLUX adapters run
            8-128; smaller ones are zero-padded into the slot.
        max_cpu_loras: Adapters held in host memory.
        lora_ckpt_paths: ``{name: path}`` declared up front, or None.
        dynamic_multi_lora: Allow adapters to be added at runtime.
        enable_base_model_only: Reserve a slot for the unmodified model.
        **kwargs: Passed through to ``LoraServingConfig``.

    Returns:
        The config, with ``target_modules`` set to the FLUX module set,
        ``lora_shard_linear_layer`` forced off, and ``is_context_encoding`` on --
        diffusion has no prefill/decode split, so every call selects its slot the
        way the LLM path does at prefill.
    """
    return FluxLoraServingConfig(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras,
        lora_ckpt_paths=lora_ckpt_paths,
        dynamic_multi_lora=dynamic_multi_lora,
        enable_base_model_only=enable_base_model_only,
        target_modules=list(FLUX_LORA_TARGET_MODULES),
        lora_shard_linear_layer=False,
        is_context_encoding=True,
        **kwargs,
    )
