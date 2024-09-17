import subprocess

import numpy as np
import json
from polygraphy import json as pjson


def backbone_save_json_parser():
    # shell command: polygraphy run deploy/onnx/sparse4dbackbone.onnx --onnxrt --onnx-outputs mark all --save-results=backbone_onnx_out.json
    subprocess.run(
        [
            "polygraphy",
            "run",
            "deploy/onnx/sparse4dbackbone.onnx",
            "--onnxrt",
            "--onnx-outputs",
            "mark",
            "all",
            "--save-results=backbone_onnx_out.json",
        ],
        check=True,
    )

    # shell command: polygraphy run deploy/onnx/sparse4dbackbone.onnx --trt --trt-outputs mark all --save-results=backbone_trt_out.json
    subprocess.run(
        [
            "polygraphy",
            "run",
            "deploy/onnx/sparse4dbackbone.onnx",
            "--trt",
            "--trt-outputs",
            "mark",
            "all",
            "--save-results=backbone_trt_out.json",
        ],
        check=True,
    )

    f = open("backbone_onnx_out.json")
    info_onnx = json.load(f)
    f = open("backbone_trt_out.json")
    info_trt = json.load(f)
    f.close()
    onnx_outputs, trt_outputs = info_onnx["lst"][0][1][0], info_trt["lst"][0][1][0]
    onnx_layers_outputs, trt_layers_outputs = (
        onnx_outputs["outputs"],
        trt_outputs["outputs"],
    )
    print(
        "onnx节点数:",
        len(onnx_layers_outputs.keys()),
        ",",
        "trt节点数:",
        len(trt_layers_outputs.keys()),
    )
    trouble_layers, ok_layers = [], []
    for layer, value in onnx_layers_outputs.items():
        if layer in trt_layers_outputs.keys():
            onnx_out = pjson.from_json(json.dumps(value)).arr
            trt_out = pjson.from_json(json.dumps(value)).arr
            print(np.size(onnx_out), np.size(trt_out), layer)
            np.testing.assert_allclose(onnx_out, trt_out, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    backbone_save_json_parser()
