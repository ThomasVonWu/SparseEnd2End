# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import math
import json
import copy
import pyquaternion
import numpy as np


from typing import Union, List, Tuple, Dict
from torch.utils.data import Dataset
from dataset.pipeline.nuscenes import *

from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs


class NuScenes4DDetTrackDataset(Dataset):
    def __init__(
        self,
        classes: Union[str, Tuple],
        ann_file: Union[str, List],
        data_root: str,
        pipeline: Dict,
        data_aug_conf=None,
        use_valid_flag=False,
        load_interval=1,
        with_velocity=True,
        with_seq_flag=False,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
        train_mode=False,
        test_mode=False,
        val_mode=False,
        version="v1.0-mini",
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
    ) -> None:
        super().__init__()

        self._classes = classes
        self._ann_file = ann_file
        self._data_root = data_root
        self._pipeline = self.compose(pipeline)
        self._data_aug_conf = data_aug_conf
        self._use_valid_flag = use_valid_flag
        self._load_interval = load_interval
        self._with_velocity = with_velocity
        self._sequences_split_num = sequences_split_num
        self._test_mode = test_mode

        self._keep_consistent_seq_aug = keep_consistent_seq_aug
        self._ordered_annotations = self.load_annotations(ann_file)
        self._train_mode = train_mode
        self._test_mode = test_mode
        self._val_mode = val_mode

        if with_seq_flag:
            self._set_sequence_group_flag()

        self._version = version
        self._errnamemapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        self._defaultattribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.moving",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": "",
        }
        self._tracking = tracking
        self._tracking_threshold = tracking_threshold
        self._det3d_eval_version = det3d_eval_version
        self._det3d_eval_configs = det_configs(self._det3d_eval_version)
        self._track3d_eval_version = track3d_eval_version
        self._track3d_eval_configs = track_configs(self._track3d_eval_version)

    def load_annotations(self, ann_file: Union[str, List]):
        annotations = []
        assert isinstance(
            ann_file, (str, list)
        ), f"{ann_file} is not supported for annotations loading."
        if isinstance(ann_file, str):  # data str knowned dir
            files = os.listdir(ann_file)
            ann_file = [os.path.join(ann_file, file) for file in files]
        for ann in ann_file:
            annotations += json.load(open(ann, "r"))
        ordered_annotations = list(sorted(annotations, key=lambda e: e["timestamp"]))
        ordered_annotations = ordered_annotations[:: self._load_interval]
        self._scene = list(set([ann["scene_token"] for ann in ordered_annotations]))
        return ordered_annotations

    def get_data_info(self, index):
        """format data dict fed to pipeline.
        img_filename: List[str] length=6(v)
        """

        info = self._ordered_annotations[index]

        input_dict = dict(
            sample_scene=info["scene_token"],
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,  # 单位为秒
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])

        input_dict["lidar2global"] = ego2global @ lidar2ego

        image_paths = []
        lidar2img_rts = []
        cam_intrinsic = []
        for cam_type, cam_info in info["cams"].items():
            image_paths.append(cam_info["data_path"])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(np.array(cam_info["sensor2lidar_rotation"]))
            lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T  # todo

            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = copy.deepcopy(np.array(cam_info["cam_intrinsic"]))
            cam_intrinsic.append(intrinsic)
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsic,
            )
        )

        if not self._test_mode:
            annos = self.get_ann_info(index)
            input_dict.update(annos)

        return input_dict

    def get_ann_info(self, index):
        info = self._ordered_annotations[index]
        if self._use_valid_flag:
            mask = np.array(info["valid_flag"])
        else:
            # 判断每个3Dbox里面是否都有lidar点，只保留box内存在点的3D框.
            mask = np.array(info["num_lidar_pts"]) > 0
        gt_bboxes_3d = np.array(info["gt_boxes"]).reshape(-1, 7)[mask]
        gt_names_3d = np.array(info["gt_names"])[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self._classes:
                gt_labels_3d.append(self._classes.index(cat))
            else:
                gt_labels_3d.append(-1)  # 非roi类别为无效类别，设置为-1
        gt_labels_3d = np.array(gt_labels_3d)

        if self._with_velocity:
            gt_velocity = np.array(info["gt_velocity"]).reshape(-1, 2)[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])  # 判断标注文件中的速度是否都有效
            gt_velocity[nan_mask] = [0.0, 0.0]  # 标注中速度无效设置为0
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:  # track id
            track_id = np.array(info["instance_inds"])[mask]
            anns_results["track_id"] = track_id
        return anns_results

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group.
        Return:
            self._flag: total samples group id. numpy.array(nums, dtype=np.int64)
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self._ordered_annotations)):
            if idx != 0 and len(self._ordered_annotations[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self._flag = np.array(res, dtype=np.int64)

        if self._sequences_split_num != 1:
            if self._sequences_split_num == "all":
                self._flag = np.array(
                    range(len(self._ordered_annotations)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self._flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag] / self._sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self._flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self._flag)) * self._sequences_split_num
                )
                self._flag = np.array(new_flags, dtype=np.int64)

    def compose(self, pipeline_cfg):
        transforms = list()
        for pipe in pipeline_cfg:
            args = pipe.copy()
            type = args.pop("type")
            transform = eval(type)(**args)
            transforms.append(transform)
        return transforms

    def get_augmentation(self):
        if self._data_aug_conf is None:
            return None
        H, W = self._data_aug_conf["H"], self._data_aug_conf["W"]
        fH, fW = self._data_aug_conf["final_dim"]
        # if not self.test_mode:
        if self._train_mode:
            resize = np.random.uniform(*self._data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self._data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self._data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self._data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self._data_aug_conf["rot3d_range"])
        else:  # test and val mode share the same augmentation
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self._data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def evaluate(
        self,
        results,
        metric=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        pipeline=None,
    ):
        for metric in ["detection", "tracking"]:
            tracking = metric == "tracking"
            if tracking and not self._tracking:
                continue
            result_files, tmp_dir = self._format_results(
                results, jsonfile_prefix, tracking=tracking
            )

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    ret_dict = self._evaluate_single(
                        result_files[name], tracking=tracking
                    )
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files, tracking=tracking)
            if tmp_dir is not None:
                tmp_dir.cleanup()

        return results_dict

    def _format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            import tempfile

            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = os.path.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_, tracking=tracking)}
                )
        # print(results[0])
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self._classes

        print("Start to convert detection format...")
        from tqdm import tqdm

        for sample_id, det in enumerate(tqdm(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self._tracking_threshold if tracking else None
            )
            sample_token = self._ordered_annotations[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self._ordered_annotations[sample_id],
                boxes,
                mapped_class_names,
                self._det3d_eval_configs,
                self._det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        # print(name)
                        attr = self._defaultattribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self._defaultattribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": dict(use_camera=True),
            "results": nusc_annos,
        }
        os.makedirs(jsonfile_prefix, exist_ok=True)
        res_path = os.path.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        with open(res_path, "w") as f:
            json.dump(nusc_submissions, f, indent=True, ensure_ascii=False)
        return res_path

    def _evaluate_single(self, result_path, result_name="img_bbox", tracking=False):
        from nuscenes import NuScenes

        output_dir = os.path.join(*os.path.split(result_path)[:-1])
        nusc = NuScenes(version=self._version, dataroot=self._data_root, verbose=False)
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self._det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self._version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            with open(os.path.join(output_dir, "metrics_summary.json"), "r") as f:
                metrics = json.load(f)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self._classes:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}".format(metric_prefix, self._errnamemapping[k])] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self._track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self._version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self._version,
                nusc_dataroot=self._data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            with open(os.path.join(output_dir, "metrics_summary.json"), "r") as f:
                metrics = json.load(f)
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def __getitem__(self, index):
        if isinstance(index, dict):
            aug_config = index["aug_config"]
            index = index["idx"]
        else:
            aug_config = self.get_augmentation()
        data_dict = self.get_data_info(index)
        data_dict["aug_config"] = aug_config
        for transform in self._pipeline:
            data_dict = transform(data_dict)
        return data_dict

    def __len__(self):
        return len(self._ordered_annotations)


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "track_ids" in detection:
        ids = detection["track_ids"]  # numpy
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        from nuscenes.utils.data_classes import Box as NuScenesBox

        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "track_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


if __name__ == "__main__":
    pass
