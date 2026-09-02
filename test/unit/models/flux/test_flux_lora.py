# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Unit tests for the FLUX LoRA key mapping.

The mapping tests run on CPU with synthetic tensors, so they need neither a
device nor a checkpoint. The one test that checks the converted names against the
real NxDI module tree needs a FLUX checkpoint and skips without
``FLUX_LORA_TEST_CHECKPOINT``.
"""

import os
from unittest import TestCase, main, skipIf

import torch

from neuronx_distributed_inference.models.diffusers.flux.lora import (
    FLUX_LORA_TARGET_MODULES,
    convert_flux_lora_to_nxdi,
    split_single_block_proj_out,
)

CHECKPOINT = os.environ.get("FLUX_LORA_TEST_CHECKPOINT")

ATTN_DIM = 3072          # FLUX hidden size
MLP_DIM = 4 * ATTN_DIM   # single block's mlp_hidden_dim
RANK = 8


def diffusers_key(module_path, which="A"):
    return f"transformer.{module_path}.lora_{which}.weight"


def make_pair(in_features, out_features=ATTN_DIM, rank=RANK, dtype=torch.float32):
    """A (lora_A, lora_B) pair shaped the way diffusers stores them."""
    return (
        torch.randn(rank, in_features, dtype=dtype),
        torch.randn(out_features, rank, dtype=dtype),
    )


class TestSingleBlockProjOutSplit(TestCase):
    """NxDI splits diffusers' single-block proj_out in two; the split is exact."""

    @staticmethod
    def _apply(lora_a, lora_b, x):
        return (lora_b @ (lora_a @ x.T)).T

    def _split_and_compare(self, dtype, rtol, atol):
        lora_a, lora_b = make_pair(ATTN_DIM + MLP_DIM, dtype=dtype)
        attn_half, mlp_half = split_single_block_proj_out(lora_a, lora_b, ATTN_DIM)

        x = torch.randn(4, ATTN_DIM + MLP_DIM, dtype=dtype)
        expected = self._apply(lora_a, lora_b, x)
        actual = self._apply(
            attn_half["lora_A"], attn_half["lora_B"], x[:, :ATTN_DIM]
        ) + self._apply(mlp_half["lora_A"], mlp_half["lora_B"], x[:, ATTN_DIM:])
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_split_is_algebraically_exact(self):
        """In float64 the identity holds to rounding: the split loses nothing."""
        self._split_and_compare(torch.float64, rtol=1e-12, atol=1e-12)

    def test_split_holds_in_float32(self):
        """Same identity in fp32.

        Looser on purpose. The two sides sum a 15360-wide dot product in a
        different order, so they differ by fp32 accumulation noise -- ~1e-3
        absolute on unit-normal inputs. That is float behaviour, not a defect in
        the split, which is why the exactness claim above is made in float64.
        """
        self._split_and_compare(torch.float32, rtol=1e-2, atol=5e-3)

    def test_lora_a_is_cut_and_lora_b_is_shared(self):
        lora_a, lora_b = make_pair(ATTN_DIM + MLP_DIM)
        attn_half, mlp_half = split_single_block_proj_out(lora_a, lora_b, ATTN_DIM)

        self.assertEqual(attn_half["lora_A"].shape, (RANK, ATTN_DIM))
        self.assertEqual(mlp_half["lora_A"].shape, (RANK, MLP_DIM))
        # B is duplicated, not divided: that is what makes the sum exact.
        for half in (attn_half, mlp_half):
            torch.testing.assert_close(half["lora_B"], lora_b)

    def test_rejects_a_non_concatenated_layout(self):
        lora_a, lora_b = make_pair(ATTN_DIM)  # no mlp half present
        with self.assertRaisesRegex(ValueError, "not wider than the attention half"):
            split_single_block_proj_out(lora_a, lora_b, ATTN_DIM)


class TestConvertFluxLoraToNxdi(TestCase):
    def test_strips_the_transformer_prefix(self):
        a, b = make_pair(ATTN_DIM)
        state_dict = {
            diffusers_key("transformer_blocks.0.attn.to_q", "A"): a,
            diffusers_key("transformer_blocks.0.attn.to_q", "B"): b,
        }
        out = convert_flux_lora_to_nxdi(state_dict, ATTN_DIM)
        self.assertEqual(
            sorted(out),
            [
                "transformer_blocks.0.attn.to_q.lora_A.weight",
                "transformer_blocks.0.attn.to_q.lora_B.weight",
            ],
        )
        torch.testing.assert_close(out["transformer_blocks.0.attn.to_q.lora_A.weight"], a)

    def test_passes_double_block_modules_through_unchanged(self):
        paths = [
            "transformer_blocks.3.attn.to_out.0",
            "transformer_blocks.3.attn.to_add_out",
            "transformer_blocks.3.ff.net.0.proj",
            "transformer_blocks.3.ff.net.2",
            "transformer_blocks.3.ff_context.net.2",
            "transformer_blocks.3.norm1.linear",
            "transformer_blocks.3.norm1_context.linear",
            "single_transformer_blocks.5.attn.to_v",
            "single_transformer_blocks.5.proj_mlp",
            "single_transformer_blocks.5.norm.linear",
        ]
        state_dict = {}
        for p in paths:
            a, b = make_pair(ATTN_DIM)
            state_dict[diffusers_key(p, "A")] = a
            state_dict[diffusers_key(p, "B")] = b

        out = convert_flux_lora_to_nxdi(state_dict, ATTN_DIM)
        for p in paths:
            self.assertIn(f"{p}.lora_A.weight", out)
            self.assertIn(f"{p}.lora_B.weight", out)

    def test_splits_only_the_single_block_proj_out(self):
        a_single, b_single = make_pair(ATTN_DIM + MLP_DIM)
        # The top-level proj_out is a different module and must not be split.
        a_top, b_top = make_pair(ATTN_DIM, out_features=64)
        state_dict = {
            diffusers_key("single_transformer_blocks.7.proj_out", "A"): a_single,
            diffusers_key("single_transformer_blocks.7.proj_out", "B"): b_single,
            diffusers_key("proj_out", "A"): a_top,
            diffusers_key("proj_out", "B"): b_top,
        }
        out = convert_flux_lora_to_nxdi(state_dict, ATTN_DIM)

        self.assertNotIn("single_transformer_blocks.7.proj_out.lora_A.weight", out)
        for half in ("proj_out_attn", "proj_out_mlp"):
            self.assertIn(f"single_transformer_blocks.7.{half}.lora_A.weight", out)
            self.assertIn(f"single_transformer_blocks.7.{half}.lora_B.weight", out)
        self.assertIn("proj_out.lora_A.weight", out)
        torch.testing.assert_close(out["proj_out.lora_A.weight"], a_top)

    def test_rejects_a_half_present_proj_out(self):
        a, _ = make_pair(ATTN_DIM + MLP_DIM)
        state_dict = {diffusers_key("single_transformer_blocks.0.proj_out", "A"): a}
        with self.assertRaisesRegex(ValueError, "both lora_A and lora_B"):
            convert_flux_lora_to_nxdi(state_dict, ATTN_DIM)

    def test_every_converted_module_is_a_wrappable_target(self):
        """Anything the converter emits must be covered by target_modules.

        Otherwise the module is never wrapped and its weights are silently
        dropped at load time.
        """
        paths = [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.net.0.proj",
            "transformer_blocks.0.norm1_context.linear",
            "single_transformer_blocks.0.attn.to_k",
            "single_transformer_blocks.0.proj_mlp",
            "single_transformer_blocks.0.norm.linear",
        ]
        state_dict = {}
        for p in paths:
            a, b = make_pair(ATTN_DIM)
            state_dict[diffusers_key(p, "A")] = a
            state_dict[diffusers_key(p, "B")] = b
        a, b = make_pair(ATTN_DIM + MLP_DIM)
        state_dict[diffusers_key("single_transformer_blocks.0.proj_out", "A")] = a
        state_dict[diffusers_key("single_transformer_blocks.0.proj_out", "B")] = b

        out = convert_flux_lora_to_nxdi(state_dict, ATTN_DIM)
        for key in out:
            module = key.rsplit(".lora_", 1)[0]
            self.assertTrue(
                any(module.endswith(s) for s in FLUX_LORA_TARGET_MODULES),
                f"{module} is not covered by FLUX_LORA_TARGET_MODULES",
            )


@skipIf(CHECKPOINT is None, "set FLUX_LORA_TEST_CHECKPOINT to a FLUX checkpoint")
class TestMappingAgainstRealModuleTree(TestCase):
    """Do the converted names name modules that actually exist in NxDI?

    Builds the backbone on the meta device under a mocked distributed world, so
    it costs no HBM and no compilation.
    """

    def test_converted_names_exist_in_the_backbone(self):
        from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
        from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
        from neuronx_distributed.trace.mock_torchdist import mock_distributed

        from neuronx_distributed_inference.models.diffusers.flux.application import (
            create_flux_config,
        )
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
            NeuronFluxTransformer2DModel,
        )

        tp = 4
        with mock_distributed(world_size=tp):
            torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp)
            initialize_model_parallel(tensor_model_parallel_size=tp, skip_collective_init=True)
            _, _, backbone_config, _ = create_flux_config(
                CHECKPOINT, tp, tp, torch.bfloat16, 1024, 1024
            )
            with torch.device("meta"):
                model = NeuronFluxTransformer2DModel(backbone_config)

        linears = {
            name
            for name, module in model.named_modules()
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear, torch.nn.Linear))
        }
        attn_dim = backbone_config.num_attention_heads * backbone_config.attention_head_dim

        # Synthesize an adapter covering every target module of every block.
        state_dict = {}
        for prefix, count in (
            ("transformer_blocks", backbone_config.num_layers),
            ("single_transformer_blocks", backbone_config.num_single_layers),
        ):
            for i in range(count):
                for suffix in FLUX_LORA_TARGET_MODULES:
                    path = f"{prefix}.{i}.{suffix}"
                    if path.replace(".0.", f".{i}.") not in linears and path not in linears:
                        continue
                    a, b = make_pair(attn_dim, out_features=attn_dim)
                    state_dict[diffusers_key(path, "A")] = a
                    state_dict[diffusers_key(path, "B")] = b

        self.assertTrue(state_dict, "no target module matched the backbone at all")
        out = convert_flux_lora_to_nxdi(state_dict, attn_dim)
        missing = sorted(
            {k.rsplit(".lora_", 1)[0] for k in out} - linears
        )
        self.assertEqual(missing, [], f"{len(missing)} converted names do not exist")


if __name__ == "__main__":
    main()
