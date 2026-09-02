# Standard Library
import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from neuronx_distributed_inference.modules.lora_serving.lora_checkpoint import LoraCheckpoint

WEIGHT_KEY = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"


def write_adapter(
    directory,
    weight_filename="adapter_model.safetensors",
    lora_alpha=16,
    extra_files=(),
    write_config=True,
):
    """Lay out an adapter folder on disk.

    Args:
        directory: Folder to populate.
        weight_filename: Name to save the LoRA weights under. ``None`` writes no
            weights at all.
        lora_alpha: Value recorded in ``adapter_config.json``.
        extra_files: Additional filenames to drop beside the adapter, each holding
            a pickled object that is *not* a state dict -- the shape of what a
            Trainer leaves behind.
        write_config: Whether to write ``adapter_config.json``.

    Returns:
        The tensor saved as the adapter's weights, for comparison.
    """
    os.makedirs(directory, exist_ok=True)
    if write_config:
        with open(os.path.join(directory, "adapter_config.json"), "w") as handle:
            json.dump({"lora_alpha": lora_alpha, "r": 8, "use_rslora": False}, handle)

    weights = None
    if weight_filename is not None:
        weights = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        target = os.path.join(directory, weight_filename)
        if weight_filename.endswith(".safetensors"):
            save_file({WEIGHT_KEY: weights}, target)
        else:
            torch.save({WEIGHT_KEY: weights}, target)

    for filename in extra_files:
        # Deliberately not a state dict, and not weights_only-loadable.
        torch.save(
            {"per_device_train_batch_size": 4, "fn": os.path.basename},
            os.path.join(directory, filename),
        )
    return weights


class TestLoraCheckpointFolderLoading(unittest.TestCase):
    """Selecting the weight file inside an adapter folder."""

    def setUp(self):
        self.checkpoint = LoraCheckpoint(None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, directory):
        return self.checkpoint._load_lora_state_dict_from_path(directory)

    def test_loads_safetensors_adapter(self):
        path = os.path.join(self.tmpdir, "plain")
        expected = write_adapter(path, lora_alpha=32)

        lora_scaling, state_dict = self._load(path)

        self.assertEqual(lora_scaling, (32, False))
        self.assertEqual(len(state_dict), 1)
        torch.testing.assert_close(state_dict[WEIGHT_KEY], expected)

    def test_ignores_trainer_artifacts(self):
        """training_args.bin beside the adapter must not be read as weights.

        This is the reported failure: a Trainer output directory carries
        training_args.bin, the loader treated any .bin as the checkpoint, and
        torch.load rejected it under weights_only=True.
        """
        path = os.path.join(self.tmpdir, "trainer_output")
        expected = write_adapter(
            path,
            extra_files=("training_args.bin", "optimizer.pt", "scheduler.pt"),
        )

        lora_scaling, state_dict = self._load(path)

        self.assertEqual(lora_scaling, (16, False))
        torch.testing.assert_close(state_dict[WEIGHT_KEY], expected)

    def test_prefers_safetensors_over_bin(self):
        """Both PEFT formats present: the safetensors one wins, deterministically.

        This does not reliably fail on the old code -- which file it picked
        depended on os.listdir order, and that nondeterminism is the bug.
        """
        path = os.path.join(self.tmpdir, "both")
        expected = write_adapter(path)
        # A .bin holding different values; picking it would be silently wrong.
        torch.save({WEIGHT_KEY: torch.zeros(2, 3)}, os.path.join(path, "adapter_model.bin"))

        _, state_dict = self._load(path)

        torch.testing.assert_close(state_dict[WEIGHT_KEY], expected)

    def test_accepts_custom_weight_filename(self):
        """An adapter saved under a non-PEFT name still loads."""
        path = os.path.join(self.tmpdir, "custom")
        expected = write_adapter(path, weight_filename="my_lora.safetensors")

        _, state_dict = self._load(path)

        torch.testing.assert_close(state_dict[WEIGHT_KEY], expected)

    def test_custom_filename_is_chosen_deterministically(self):
        """Several custom-named checkpoints: same choice on every call."""
        path = os.path.join(self.tmpdir, "ambiguous")
        write_adapter(path, weight_filename="b_lora.safetensors")
        write_adapter(path, weight_filename="a_lora.safetensors", write_config=False)

        chosen = {self.checkpoint._find_adapter_weight_file(path) for _ in range(5)}

        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen.pop().endswith("a_lora.safetensors"))

    def test_raises_when_folder_has_no_weights(self):
        path = os.path.join(self.tmpdir, "empty")
        write_adapter(path, weight_filename=None, extra_files=("training_args.bin",))

        with self.assertRaisesRegex(ValueError, "No valid LoRA adapter checkpoint"):
            self._load(path)

    def test_missing_adapter_config_leaves_scaling_unset(self):
        path = os.path.join(self.tmpdir, "no_config")
        write_adapter(path, write_config=False)

        lora_scaling, state_dict = self._load(path)

        self.assertIsNone(lora_scaling)
        self.assertEqual(len(state_dict), 1)

    def test_rejects_missing_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "Invalid checkpoint path"):
            self._load(os.path.join(self.tmpdir, "does_not_exist"))


class TestLoraCheckpointFileLoading(unittest.TestCase):
    """Passing a checkpoint file directly rather than a folder."""

    def setUp(self):
        self.checkpoint = LoraCheckpoint(None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_safetensors_file(self):
        path = os.path.join(self.tmpdir, "adapter")
        expected = write_adapter(path)

        _, state_dict = self.checkpoint._load_lora_state_dict_from_path(
            os.path.join(path, "adapter_model.safetensors")
        )

        torch.testing.assert_close(state_dict[WEIGHT_KEY], expected)

    def test_rejects_unknown_extension(self):
        path = os.path.join(self.tmpdir, "notes.txt")
        with open(path, "w") as handle:
            handle.write("not a checkpoint")

        with self.assertRaisesRegex(FileNotFoundError, "Invalid checkpoint filename"):
            self.checkpoint._load_lora_state_dict_from_path(path)


if __name__ == "__main__":
    unittest.main()
