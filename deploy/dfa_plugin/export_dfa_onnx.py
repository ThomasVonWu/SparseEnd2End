# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.onnx
import torch.nn as nn

import numpy as np

import os
import onnx
import onnxsim

from modules.ops.deformable_aggregation import DeformableAggregationFunction


class CustomDFAModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        value,
        input_spatial_shapes,
        input_level_start_index,
        sampling_locations,
        attention_weights,
    ):
        output = DeformableAggregationFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def export_onnx(model, save_path):
    num_cams = 6
    dummy_value = torch.rand(
        [1, num_cams * (64 * 176 + 32 * 88 + 16 * 44 + 8 * 22), 256]
    ).cuda()
    dummy_spatial_shapes = (
        torch.tensor([[64, 176], [32, 88], [16, 44], [8, 22]])
        .int()
        .unsqueeze(0)
        .repeat(num_cams, 1, 1)
        .cuda()
    )
    dummy_level_start_index = (
        torch.tensor(
            [
                [0, 11264, 14080, 14784],
                [14960, 26224, 29040, 29744],
                [29920, 41184, 44000, 44704],
                [44880, 56144, 58960, 59664],
                [59840, 71104, 73920, 74624],
                [74800, 86064, 88880, 89584],
            ]
        )
        .int()
        .cuda()
    )
    dummy_sampling_loc = torch.rand([1, 900, 13, 6, 2]).cuda()
    dummy_weights = torch.rand([1, 900, 13, 6, 4, 8]).cuda()

    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=(
                dummy_value,
                dummy_spatial_shapes,
                dummy_level_start_index,
                dummy_sampling_loc,
                dummy_weights,
            ),
            f=save_path,
            input_names=[
                "value",
                "spatial_shapes",
                "level_start_index",
                "sampling_loc",
                "attn_weight",
            ],
            output_names=["output0"],
            opset_version=15,
            do_constant_folding=True,
            verbose=False,
        )

    # check the exported onnx model
    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, save_path)
    print(
        f"[Succed] Simplifying with onnx-simplifier {onnxsim.__version__}, file is saved in {save_path}."
    )

    # inference results
    output = model(
        dummy_value,
        dummy_spatial_shapes,
        dummy_level_start_index,
        dummy_sampling_loc,
        dummy_weights,
    )

    print("dummy_value.dtype=", dummy_value.dtype)
    print("dummy_spatial_shapes.dtype=", dummy_spatial_shapes.dtype)
    print("dummy_level_start_index.dtype=", dummy_level_start_index.dtype)
    print("dummy_sampling_loc.dtype=", dummy_sampling_loc.dtype)
    print("dummy_weights.dtype=", dummy_weights.dtype)

    save_bin(
        dummy_value.cpu().numpy(),
        dummy_spatial_shapes.cpu().numpy(),
        dummy_level_start_index.cpu().numpy(),
        dummy_sampling_loc.cpu().numpy(),
        dummy_weights.cpu().numpy(),
        output.cpu().numpy(),
    )
    print("[output]:\n", output)
    print("[output_shape]=", output.shape)
    print("[output_sum]  =", output.cpu().numpy().sum())
    print("[output_max]  =", torch.max(output), "\n[output_min]=", torch.min(output))


def save_bin(*data, prefix="deploy/dfa_plugin/asset/"):
    os.makedirs(prefix, exist_ok=True)

    (
        dummy_value,
        dummy_spatial_shapes,
        dummy_level_start_index,
        dummy_sampling_loc,
        dummy_weights,
        output,
    ) = data

    dummy_value.tofile(prefix + "value.bin")
    dummy_spatial_shapes.tofile(prefix + "spatial_shapes.bin")
    dummy_level_start_index.tofile(prefix + "level_start_index.bin")
    dummy_sampling_loc.tofile(prefix + "sampling_loc.bin")
    dummy_weights.tofile(prefix + "attn_weight.bin")
    output.tofile(prefix + "output.bin")
    dummy_spatial_shapes.astype(np.float32).tofile(prefix + "spatial_shapes_f.bin")
    dummy_level_start_index.astype(np.float32).tofile(
        prefix + "level_start_index_f.bin"
    )
    print("[Success] Save InputandOutput BinFile!")


if __name__ == "__main__":
    setup_seed(1)
    model = CustomDFAModel()
    model.eval()
    save_path = "deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx"
    export_onnx(model, save_path)
