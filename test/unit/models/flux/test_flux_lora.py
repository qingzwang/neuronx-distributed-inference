# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Unit tests for FLUX LoRA: the key mapping and the adapter selection logic.

These run on CPU with synthetic tensors, so they need neither a device nor a
checkpoint. The one test that checks the converted names against the real NxDI
module tree needs a FLUX checkpoint and skips without
``FLUX_LORA_TEST_CHECKPOINT``.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main, skipIf

import torch

from neuronx_distributed_inference.models.diffusers.flux.lora import (
    FLUX_LORA_TARGET_MODULES,
    build_flux_lora_config,
    convert_flux_lora_to_nxdi,
    split_single_block_proj_out,
)
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
    NeuronFluxBackboneApplication,
)
from neuronx_distributed_inference.modules.lora_serving.lora_model import LoraModelManager

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


class _StubBackbone:
    """Enough of NeuronFluxBackboneApplication to exercise adapter selection.

    The method under test is taken unbound from the real class, so it is the
    shipping code that runs; only its surroundings -- a device, a traced graph --
    are stubbed out. ``dynamic_update_weights_for_lora`` is replaced by a recorder
    that offsets the ids, so a test can tell whether the dynamic path was taken.
    """

    select_lora_adapters = NeuronFluxBackboneApplication.select_lora_adapters
    DYNAMIC_OFFSET = 100

    def __init__(self, lora_config):
        self.neuron_config = SimpleNamespace(lora_config=lora_config)
        self.default_adapter_ids = None
        self.dynamic_calls = []
        self.models = [
            SimpleNamespace(model=SimpleNamespace(nxd_model=SimpleNamespace(weights="W")))
        ]
        if lora_config is not None:
            self.lora_model_manager = LoraModelManager(lora_config)
            self.lora_model_manager.dynamic_update_weights_for_lora = self._record

    def _record(self, weights, ids):
        self.dynamic_calls.append((weights, ids.tolist()))
        return ids + self.DYNAMIC_OFFSET


class TestSelectLoraAdapters(TestCase):
    """Names in, device slot indices out."""

    @classmethod
    def setUpClass(cls):
        # LoraServingConfig insists the paths exist, but nothing here reads them.
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        for name in ("first", "second"):
            path = os.path.join(cls._tmp.name, f"{name}.safetensors")
            open(path, "wb").close()
            cls.paths[name] = path

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _stub(self, dynamic=False):
        return _StubBackbone(
            build_flux_lora_config(
                max_loras=2,
                max_lora_rank=RANK,
                lora_ckpt_paths=dict(self.paths),
                dynamic_multi_lora=dynamic,
            )
        )

    def test_no_lora_config_selects_nothing(self):
        self.assertIsNone(_StubBackbone(None).select_lora_adapters(["first"], 1))

    def test_names_map_to_slots_in_declaration_order(self):
        # Slot 0 is the base model, so the first declared adapter is slot 1.
        stub = self._stub()
        self.assertEqual(stub.select_lora_adapters(["first"], 1).tolist(), [1])
        self.assertEqual(stub.select_lora_adapters(["second"], 1).tolist(), [2])
        self.assertEqual(stub.dynamic_calls, [], "static path must not swap weights")

    def test_no_adapter_means_the_base_slot(self):
        self.assertEqual(self._stub().select_lora_adapters(None, 2).tolist(), [0, 0])

    def test_a_single_name_is_broadcast_across_the_batch(self):
        stub = self._stub()
        self.assertEqual(stub.select_lora_adapters("second", 2).tolist(), [2, 2])
        self.assertEqual(stub.select_lora_adapters(["second"], 2).tolist(), [2, 2])

    def test_set_lora_adapters_supplies_the_default(self):
        stub = self._stub()
        NeuronFluxBackboneApplication.set_lora_adapters(stub, "second")
        self.assertEqual(stub.select_lora_adapters(None, 1).tolist(), [2])
        NeuronFluxBackboneApplication.set_lora_adapters(stub, None)
        self.assertEqual(stub.select_lora_adapters(None, 1).tolist(), [0])

    def test_a_tensor_is_taken_as_slots_already_resolved(self):
        stub = self._stub(dynamic=True)
        ids = torch.tensor([2], dtype=torch.int32)
        self.assertIs(stub.select_lora_adapters(ids, 1), ids)
        self.assertEqual(stub.dynamic_calls, [], "an explicit slot must not be remapped")

    def test_a_slot_past_the_last_one_is_rejected(self):
        """Out of range must raise rather than reach the device.

        The graph does not bounds-check: it gathers out of bounds, floods the log
        with DGE notifications and returns whatever it read.
        """
        stub = self._stub()  # max_loras=2 + base slot = slots 0..2
        with self.assertRaisesRegex(ValueError, "beyond max_loras"):
            stub.select_lora_adapters(torch.tensor([3], dtype=torch.int32), 1)

    def test_dynamic_mode_makes_the_adapter_resident_first(self):
        stub = self._stub(dynamic=True)
        slots = stub.select_lora_adapters(["second"], 1)
        self.assertEqual(stub.dynamic_calls, [("W", [2])])
        # The returned slot is whatever the cache assigned, not the adapter id.
        self.assertEqual(slots.tolist(), [2 + _StubBackbone.DYNAMIC_OFFSET])

    def test_dynamic_mode_also_resolves_the_base_slot(self):
        stub = self._stub(dynamic=True)
        stub.select_lora_adapters(None, 1)
        self.assertEqual(stub.dynamic_calls, [("W", [0])])


class TestBuildFluxLoraConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmp.name, "adapter.safetensors")
        open(cls.path, "wb").close()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _config(self, **kwargs):
        return build_flux_lora_config(
            max_lora_rank=RANK, lora_ckpt_paths={"a": self.path}, **kwargs
        )

    def test_uses_the_flux_module_set_without_reading_adapter_config_json(self):
        # A bare .safetensors with no adapter_config.json next to it: the base
        # class would raise here.
        config = self._config()
        self.assertEqual(config.target_modules, list(FLUX_LORA_TARGET_MODULES))
        self.assertEqual(config.max_lora_rank, RANK)

    def test_keeps_lora_a_unsharded(self):
        # lora_B has to follow the base layer's gather_output, which the FLUX LoRA
        # modules handle themselves; sharding lora_A as well breaks that.
        self.assertFalse(self._config().lora_shard_linear_layer)

    def test_reserves_a_base_slot_on_top_of_the_requested_ones(self):
        config = self._config(max_loras=2, max_cpu_loras=3)
        self.assertEqual(config.max_loras, 3)
        self.assertEqual(config.max_cpu_loras, 4)

    def test_dynamic_mode_puts_declared_adapters_in_the_host_tier_too(self):
        # An adapter declared at build time must stay reloadable, otherwise it
        # cannot be swapped back in after the device cache evicts it.
        self.assertEqual(self._config(dynamic_multi_lora=True).lora_ckpt_paths_cpu,
                         {"a": self.path})
        self.assertEqual(self._config().lora_ckpt_paths_cpu, {})


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

    def test_lora_layers_report_the_dtype_their_weights_will_have(self):
        """The reported dtype has to survive the backbone's own cast.

        FLUX builds its linears in float32 and casts the backbone to torch_dtype
        afterwards, so a LoRA layer that copied its dtype from the base layer
        reports float32 while its parameters end up bfloat16. A dynamic swap
        allocates the host buffer from the reported dtype and copies it into the
        device tensor, so the two have to agree.
        """
        from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
        from neuronx_distributed.trace.mock_torchdist import mock_distributed

        from neuronx_distributed_inference.models.diffusers.flux.application import (
            create_flux_config,
        )
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
            NeuronFluxTransformer2DModel,
        )
        from neuronx_distributed_inference.modules.lora_serving.lora_layer import BaseMultiLora

        tp = 4
        with tempfile.TemporaryDirectory() as tmp:
            adapter = os.path.join(tmp, "adapter.safetensors")
            open(adapter, "wb").close()
            lora_config = build_flux_lora_config(
                max_lora_rank=RANK, lora_ckpt_paths={"a": adapter}
            )
            with mock_distributed(world_size=tp):
                torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp)
                initialize_model_parallel(
                    tensor_model_parallel_size=tp, skip_collective_init=True
                )
                _, _, backbone_config, _ = create_flux_config(
                    CHECKPOINT, tp, tp, torch.bfloat16, 1024, 1024, lora_config=lora_config
                )
                with torch.device("meta"):
                    model = NeuronFluxTransformer2DModel(backbone_config)

        layers = [m for m in model.modules() if isinstance(m, BaseMultiLora)]
        self.assertTrue(layers, "the backbone was not wrapped with LoRA at all")
        wrong = {str(m.get_weight_dtype()) for m in layers} - {"torch.bfloat16"}
        self.assertEqual(wrong, set())

        model = model.to(dtype=torch.bfloat16)
        for m in layers:
            self.assertEqual(m.weight.dtype, m.get_weight_dtype())


if __name__ == "__main__":
    main()
