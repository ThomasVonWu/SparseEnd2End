import json
import numpy as np

from polygraphy.json import save_json
from polygraphy import json as pjson

import logging
from tool.utils.logger import logger_wrapper


def first_frame_head_input_json_save(
    logger,
    input_json_path="deploy/utils/first_frame_head_inputs.json",
    i=0,
):

    prefix = "script/tutorial/asset/"
    feature_shape = [1, 89760, 256]
    feature = np.fromfile(
        f"{prefix}sample_{i}_feature_1*89760*256_float32.bin",
        dtype=np.float32,
    ).reshape(feature_shape)

    spatial_shapes_shape = [6, 4, 2]
    spatial_shapes = np.fromfile(
        f"{prefix}sample_{i}_spatial_shapes_6*4*2_int32.bin", dtype=np.int32
    ).reshape(spatial_shapes_shape)

    level_start_index_shape = [6, 4]
    level_start_index = np.fromfile(
        f"{prefix}sample_{i}_level_start_index_6*4_int32.bin",
        dtype=np.int32,
    ).reshape(level_start_index_shape)

    instance_feature_shape = [1, 900, 256]
    instance_feature = np.fromfile(
        f"{prefix}sample_{i}_instance_feature_1*900*256_float32.bin",
        dtype=np.float32,
    ).reshape(instance_feature_shape)

    anchor_shape = [1, 900, 11]
    anchor = np.fromfile(
        f"{prefix}sample_{i}_anchor_1*900*11_float32.bin", dtype=np.float32
    ).reshape(anchor_shape)

    time_interval_shape = [1]
    time_interval = np.fromfile(
        f"{prefix}sample_{i}_time_interval_1_float32.bin",
        dtype=np.float32,
    ).reshape(time_interval_shape)

    image_wh_shape = [1, 6, 2]
    image_wh = np.fromfile(
        f"{prefix}sample_{i}_image_wh_1*6*2_float32.bin",
        dtype=np.float32,
    ).reshape(image_wh_shape)

    lidar2img_shape = [1, 6, 4, 4]
    lidar2img = np.fromfile(
        f"{prefix}sample_{i}_lidar2img_1*6*4*4_float32.bin",
        dtype=np.float32,
    ).reshape(lidar2img_shape)

    save_json(
        [
            {
                "feature": feature,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "instance_feature": instance_feature,
                "anchor": anchor,
                "time_interval": time_interval,
                "image_wh": image_wh,
                "lidar2img": lidar2img,
            },
        ],
        input_json_path,
    )


def run_polygraphy_shell_command(output_json_path: str = ""):
    import subprocess

    # subprocess.run(
    #     [
    #         "polygraphy",
    #         "run",
    #         "deploy/onnx/sparse4dhead1st.onnx",
    #         "--trt",
    #         "--verbose",
    #         "--load-inputs=deploy/utils/first_frame_head_inputs.json",
    #         "--trt-outputs",
    #         "mark",
    #         "all",
    #         "--save-results=" + output_json_path,
    #         "--plugins",
    #         "deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    #     ],
    #     check=True,
    # )

    subprocess.run(
        [
            "polygraphy",
            "run",
            "deploy/onnx/sparse4dhead1st.onnx",
            "--trt",
            "--verbose",
            "--load-inputs=deploy/utils/first_frame_head_inputs.json",
            "--save-results=" + output_json_path,
            "--plugins",
            "deploy/dfa_plugin/lib/deformableAttentionAggr.so",
        ],
        check=True,
    )


def parse_first_frame_head_polygraphy_output_json(
    logger,
    path="",
):
    def printArrayInformation(x, logger, info: str):
        logger.debug(f"Polygraphy: [{info}]")
        logger.debug(
            "\tMax=%.3f, Min=%.3f, SumAbs=%.3f"
            % (
                np.max(x),
                np.min(x),
                np.sum(abs(x)),
            )
        )
        logger.debug(
            "\tfirst5 | last5 %s  ......  %s" % (x.reshape(-1)[:5], x.reshape(-1)[-5:])
        )

    info_trt = json.load(open(path, "r"))
    trt_layers_outputs = info_trt["lst"][0][1][0]["outputs"]

    for layer, value in trt_layers_outputs.items():
        trt_out = pjson.from_json(json.dumps(value)).arr
        roi_layer = [
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality_score",
        ]
        if layer in roi_layer:
            printArrayInformation(trt_out, logger, layer)


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.DEBUG)

    input_json_path = "deploy/utils/first_frame_head_inputs.json"
    first_frame_head_input_json_save(logger, input_json_path, 0)
    logger.info(
        f"Step 1. Generate first frame head input json successfully, path: {input_json_path} ."
    )

    # polygraphy run deploy/onnx/sparse4dhead1st.onnx --trt --verbose --load-inputs=deploy/utils/first_frame_head_inputs.json --trt-outputs mark all --save-results=deploy/utils/first_frame_head_outputs.json  --plugins deploy/dfa_plugin/lib/deformableAttentionAggr.so
    output_json_path = "deploy/utils/first_frame_head_outputs.json"
    run_polygraphy_shell_command(output_json_path)
    logger.info(
        f"Step 2. Polygraphy export first frame head output json successfully, path: {output_json_path} ."
    )

    parse_first_frame_head_polygraphy_output_json(logger, output_json_path)
    logger.info(f"Step 3. Parse first frame head output json successfully .")
