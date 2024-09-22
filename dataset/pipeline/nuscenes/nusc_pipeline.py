# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import cv2
import numpy as np
import torch

from PIL import Image
from numpy import random
from typing import List
from dataset.utils.data_container import DataContainer


class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call function to load multi-view image from files.
           Format and norm dict keys and add new keys.

        Args:
            results (dict): Result dict from DumpAnnoJson containing multi-view image filenames.
                - sample_idx
                - pts_filename
                - sweeps
                - timestamp
                - lidar2ego_translation
                - lidar2ego_rotation
                - ego2global_translation
                - ego2global_rotation
                - lidar2global
                - img_filename => filename
                - lidar2img
                - cam_intrinsic
                - gt_bboxes_3d
                - gt_labels_3d
                - gt_names
                - instance_inds
                - aug_config

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (list(str)): Multi-view image filenames.
                - img (list(np.ndarray.uint8)): Multi-view image arrays, Shape(900, 1600, 3).
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h， w, c, num_views)(900,1600,3,6)
        img = np.stack([cv2.imread(name) for name in filename], axis=-1)  # bgr
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape  # todo
        # Set initial values for default meta_keys
        results["scale_factor"] = 1.0  # todo
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        return repr_str


class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
    """

    def __init__(
        self,
        load_dim=5,
        use_dim=[0, 1, 2],
        shift_height=False,
    ):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"

        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename: str):
        """Private function to load point clouds data.

        Args:
            - pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array with Shape(nums_pts, ) containing point clouds data.
        """
        try:
            with open(pts_filename, "rb") as f:
                pts_bytes = f.read()
            points = np.frombuffer(pts_bytes, dtype=np.float32)  # (nums_pts, )
        except ConnectionError:
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (numpy.array): Point clouds data with Shape(34400, 5).
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )

        results["points"] = points
        return results


class ResizeCropFlipImage(object):
    def __call__(self, results):
        """
        Updated keys and values are described below.
        Return:
            - img           : List(numpy.array, ...) dtype=np.float32.
            - img_shape     : List[Tuple(256, 704, 3)] length=nums_cam.
            - cam_intrinsic : List[numpy.array,...] Shape(3,3) length=nums_cam.
            - lidar2img     : List[numpy.array,...] Shape(4,4) length=nums_cam.
        """
        aug_config = results.get("aug_config", None)
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        results["ori_img"] = np.stack(
            [
                np.ascontiguousarray(img.astype(np.uint8).transpose(2, 0, 1)[::-1, ...])
                for img in imgs
            ],
            axis=0,
        )  # (6,3,900,1600) RGB
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]),
                aug_config,
            )
            new_imgs.append(img)
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        """
        Return:
            - img : numpy.array Shape(256, 704, 3) Dtype.float32.
            - extend_matrix: numpy.array Shape(4, 4) Dtype.float64.
        """
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # PIL just for img transform
        origin_dtype = img.dtype
        assert origin_dtype == np.uint8
        """ Test resize gap between opencv with pillow.
        img_1 = cv2.resize(
            src=img.copy(), dsize=resize_dims, interpolation=cv2.INTER_LINEAR
        )
        img_2 = np.array(
            Image.fromarray(img.copy()).resize(resize_dims, Image.BILINEAR)
        )
        max_abs_err = np.max(np.abs(img_2 - img_1))
        print(max_abs_err) # 255
        """
        img = cv2.resize(src=img, dsize=resize_dims, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        # img = img.resize(resize_dims).crop(crop) # change to opencv
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]])
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=[4, 8, 16], max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth  # 根据nuscenes距离任务确定

    def __call__(self, input_dict):
        """将送入模型的原始图下采样3次, 生成depthmap
            下采样三次: num_level=3.
            - gt_depth: List[List(numpy), ...]
                外层: length=num_level
                内层: length=num_cams.
                Shape[64, 176] float32.
                Shape[32, 88]  float32.
                Shape[16, 44]  float32.
        Added keys and values are described below.
        未被投影位置默认填充-1.
            - gt_depth : List[numpy.array, ...]
                         ([6, 64, 176])
                         ([6, 32, 88])
                         ([6, 16, 44])
        """
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        gt_depth_ori = list()
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = np.squeeze(lidar2img[:3, :3] @ points, axis=-1) + lidar2img[:3, 3]
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

            h, w = (int(H), int(W))
            u = np.floor(U).astype(np.int32)
            v = np.floor(V).astype(np.int32)
            depth_map = np.zeros([h, w], dtype=np.float32)
            depth_map[v, u] = depths
            gt_depth_ori.append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        input_dict["gt_depth_ori"] = np.stack(gt_depth_ori, axis=0)
        return input_dict


class BBoxRotation(object):
    def __call__(self, results):
        """
        BEV augmentation, updated keys and values are described below.
        Return:
            - lidar2global : numpy.array Shape(4,4) dtype=np.float64.
            - gt_bboxes_3d : numpy.array Shape(nums_bbox, 9) dtype=np.float64.
        """
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array([[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]])
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
            updated keys and values are described below.
            - img : List(numpy.array, ...) Shape(256,704,3) dtype=np.float32.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast last
            # mode == 1 --> do random contrast first
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color fromBGR to HSV
            code = getattr(cv2, "COLOR_BGR2HSV")
            img = cv2.cvtColor(img, code)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            code = getattr(cv2, "COLOR_HSV2BGR")
            img = cv2.cvtColor(img, code)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)  # (3,)
        self.std = np.array(std, dtype=np.float32)  # (3,)
        self.to_rgb = to_rgb

    def imnormalize_(self, img, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        img = img.copy()  # (256,704,3)
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace (float64-float64 or float32-float64)
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
            if self.to_rgb True[default], 'img' key is updated:
                bgr to rgb
        """
        results["img"] = [
            self.imnormalize_(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


class CircleObjectRangeFilter(object):
    def __init__(self, class_dist_thred=[55] * 10):
        self.class_dist_thred = (
            class_dist_thred  # 长度需要和task classes类别长度保持一致
        )

    def __call__(self, input_dict):
        """
        对应类别设置距离限制, 过滤满足条件的gtbboxes:
        Args:
            gt_bboxes_3d : (nums_ori, 9)
            gt_labels_3d : (nums_ori, )
        Return:
            gt_bboxes_3d : (nums_filtered 9)
            gt_labels_3d : (nums_filtered,)
            track_id     : (nums_filtered,)
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1))
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "track_id" in input_dict:
            input_dict["track_id"] = input_dict["track_id"][mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


class NuScenesSparse4DAdaptor(object):
    """global2lidar and lidar2gloab 的类型还是float64, 类型会对后面产生影响吗？
    Add
        - image_wh:  list[(int(h), int(w)),...] => numpy.array.shape(6,2) dtype=float32.
        - global2lidar : numpy.array.shape(4,4) dtype=np.float64.
        - focal: (nums_cam,).
    Updated keys and values are described below.
        - lidar2img : list[numpy.array.shape(4,4),...] dtype=float64 => numpy.array.shape(6,4,4) dtype=float32.
        - cam_intrinsic : list[numpy.array.shape(3,3),...] dtype=float64 => numpy.array.shape(6,3,3) dtype=float32.
        - gt_bboxes_3d : to DataContainer (train or val), torch.tensor.shape(nums_gt, 9) dtype=torch.float32 device=cpu.
        - gt_labels_3d : to DataContainer (train or val), torch.tensor.shape(nums_gt, ) dtype=torch.int64 device=cpu.
        - img : [h,w,c] => [c,h,w]
                DataContainer (stack = True), torch.tensor.shape(num_imgs, 3, h, w) dtype=torch.float32.
    """

    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["lidar2img"] = np.stack(input_dict["lidar2img"]).astype(np.float32)
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, ::-1]
        )  # (h,w) => (w,h)
        input_dict["global2lidar"] = np.linalg.inv(input_dict["lidar2global"])
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.stack(input_dict["cam_intrinsic"]).astype(
                np.float32
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DataContainer(
                torch.from_numpy(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DataContainer(
                torch.from_numpy(input_dict["gt_labels_3d"]).long()
            )
        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DataContainer(torch.from_numpy(imgs), stack=True)
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


class Collect:
    """Collect data from the loader relevant to the specific task.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``DataContainer`` and collected in ``data[img_metas]``.
    """

    def __init__(
        self,
        keys,
        meta_keys=("filename", "lidar2global", "global2lidar", "timestamp", "track_id"),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :object DataContainer.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_metas"] = DataContainer(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )
