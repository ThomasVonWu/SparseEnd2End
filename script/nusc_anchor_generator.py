# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import json
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from dataset.config.nusc_std_bbox3d import *


def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    data = list()
    for ann in ann_file:
        data += json.load(open(ann, "r"))
    gt_boxes_list = list()
    for x in tqdm(data, desc="Process Samples:"):
        if len(x["gt_boxes"]) != 0:
            gt_boxes_list.append(x["gt_boxes"])
        else:
            print("[WARNING] Gt boxes is empty!")
            continue
    gt_boxes = np.concatenate(gt_boxes_list, axis=0)
    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    mask = distance <= detection_range
    gt_boxes = gt_boxes[mask]
    # clf = KMeans(n_clusters=num_anchor, verbose=verbose, n_init="auto")
    clf = KMeans(n_clusters=num_anchor, verbose=verbose)
    print("===========Starting kmeans, please wait.===========")
    clf.fit(gt_boxes[:, [X, Y, Z]])
    anchor = np.zeros((num_anchor, 11))
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    anchor[:, COS_YAW] = 1
    np.save(output_file_name, anchor)
    print(f"===========Done! Save results to {output_file_name}.===========")
    print("Check GT Consistency:")
    print(gt_boxes.shape)
    print(gt_boxes[:, [X, Y, Z]].mean(axis=0))
    print(gt_boxes[:, [W, L, H]].mean(axis=0))
    print(gt_boxes[:, [SIN_YAW]].mean(axis=0))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nuscenes anchor kmeans")
    parser.add_argument("--ann_file", type=str, default=None)
    parser.add_argument(
        "--ann_file_dir", type=str, default="data/nusc_anno_dumpjson/train"
    )
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="data/nuscenes_kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.ann_file_dir is not None:
        ann_names = os.listdir(args.ann_file_dir)
        anns_path = [os.path.join(args.ann_file_dir, ann) for ann in ann_names]

    elif args.ann_file is not None:
        anns_path = [args.ann_file]

    get_kmeans_anchor(
        anns_path,
        args.num_anchor,
        args.detection_range,
        args.output_file_name,
        args.verbose,
    )
