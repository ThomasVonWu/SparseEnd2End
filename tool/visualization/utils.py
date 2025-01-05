# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import copy

import cv2
import numpy as np
import torch

from tqdm import tqdm
from dataset.config.nusc_std_bbox3d import *
from typing import Dict, Tuple, List, Union


def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child": (135, 206, 235),  # Skyblue,
        "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
        "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
        "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
        "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (210, 105, 30),  # Chocolate
        "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
        "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy": (255, 127, 80),  # Coral
        "vehicle.bus.rigid": (255, 69, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction": (233, 150, 70),  # Darksalmon
        "vehicle.emergency.ambulance": (255, 83, 0),
        "vehicle.emergency.police": (255, 215, 0),  # Gold
        "vehicle.motorcycle": (255, 61, 99),  # Red
        "vehicle.trailer": (255, 140, 0),  # Darkorange
        "vehicle.truck": (255, 99, 71),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other": (175, 0, 75),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (112, 180, 60),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (255, 228, 196),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245),
    }

    return classname_to_color


def get_task_colormap() -> Dict[str, Tuple[int, int, int]]:
    """RGB"""
    classname_to_color = {
        "barrier": (112, 128, 144),  # Slategrey,
        "bicycle": (220, 20, 60),  # Crimson,
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange,
        "construction_vehicle": (233, 150, 70),  # Darksalmon,
        "motorcycle": (255, 61, 99),  # Red,
        "pedestrian": (0, 0, 230),  # Blue,
        "traffic_cone": (47, 79, 79),  # Darkslategrey,
        "trailer": (255, 140, 0),  # Darkorange,
        "truck": (255, 99, 71),  # Tomato,
    }
    return classname_to_color


def get_id_class_map() -> Dict[int, str]:
    index_to_class = {
        0: "car",
        1: "truck",
        2: "construction_vehicle",
        3: "bus",
        4: "trailer",
        5: "barrier",
        6: "motorcycle",
        7: "bicycle",
        8: "pedestrian",
        9: "traffic_cone",
    }
    return index_to_class


def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


def draw_meatas(img, start_pts: tuple, track_id: int, color, fontScale=0.3):
    """
    start_pts : text start pixel coordinate.
    imgshape  : (256, 704, 3).
    """
    x0, y0 = start_pts

    # 获取文本标签的大小
    labelSize = cv2.getTextSize(
        text=f"{track_id}",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        thickness=2,
    )[0]

    # 基於角點向上移動3個像素
    start_y = int(y0 - labelSize[1] - 3)
    end_y = int(y0 - 3)

    # （左上，右下)
    cv2.rectangle(
        img=img,
        pt1=(x0, start_y),
        pt2=(x0 + labelSize[0], end_y),
        color=color,
        thickness=-1,
    )

    # 矩形框内填充黑色字体,（text的起始點位於左下方)
    cv2.putText(
        img=img,
        text=f"{track_id}",
        org=(x0, end_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(0, 0, 0),
        thickness=1,
    )
    return img


def draw_class_label(
    img, x1, y1, color, label: str, fontScale=0.3, yaw=0, with_yaw_label=False
):
    labelSize = cv2.getTextSize(
        text=label + "_{:.0f}".format(yaw / 3.14159 * 180),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        thickness=2,
    )[0]

    start_y = int(y1 - labelSize[1] - 3)
    end_y = int(y1 - 3)
    if y1 - labelSize[1] - 3 < 1:
        start_y = y1 + 2
        end_y = y1 + labelSize[1] + 3

    cv2.rectangle(
        img=img,
        pt1=(x1, start_y),
        pt2=(x1 + labelSize[0] - 10, end_y),
        color=color,
        lineType=cv2.LINE_AA,
        thickness=-1,
    )
    if with_yaw_label:
        label = (label + "_{:.0f}".format(yaw / 3.14159 * 180),)
    cv2.putText(
        img=img,
        text=label,
        org=(x1, end_y - 1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(0, 0, 0),
        thickness=1,
    )
    return img


def plot_rect3d_on_img(
    img, num_rects, rect_corners, label, img_metas, color, thickness, cam_index
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        label (numpy.array): [num_rect, ].
        color (tuple[int], optional): The color to draw bboxes.
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    color_flag = color is None
    id_class_map = get_id_class_map()
    colormap = get_task_colormap()

    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )

    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if color_flag:
                color = colormap[id_class_map[label[i]]]

            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
            img = draw_class_label(
                img,
                int((corners[2, 0] + corners[3, 0]) / 2),
                int((corners[0, 1] + corners[1, 1]) / 2),
                color=color,
                label=id_class_map[label[i]],
            )
        # cv2.imshow("val_pipeline", img[..., ::-1].astype(np.uint8))
        # key = cv2.waitKey(0)
        # if key == 27:
        #     continue

    if img_metas is not None:
        track_id = img_metas.get("track_id", None)
        if track_id is not None:
            assert len(track_id) == num_rects, f"{len(track_id)} v.s. {num_rects}."
            for i in range(num_rects):
                corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
                x0 = int((corners[2, 0] + corners[3, 0]) / 2)
                y0 = int((corners[4, 1] + corners[7, 1]) / 2)
                draw_meatas(
                    img, (x0, y0), track_id[i], colormap[id_class_map[label[i]]]
                )
                draw_legend(img, cam_index)

    return img.astype(np.uint8)


def draw_legend(img, cam_index):
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    cam_name = camera_types[cam_index]
    cv2.putText(
        img=img,
        text=f"{cam_name}",
        org=(8, 16),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(80, 127, 255),
        thickness=1,
    )
    pass


def draw_lidar_bbox3d_on_img(
    bboxes3d,
    label,
    raw_img,
    lidar2img_rt,
    img_metas=None,
    color=None,
    thickness=1,
    cam_index=0,
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:torch.tensor): (nums_bbox, 9) dtype=torch.float32.
        label (:torch.tensor): (nums_bbox, ) dtype=torch.int64.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(
        img,
        num_bbox,
        imgfov_pts_2d,
        label.detach().cpu().numpy(),
        img_metas,
        color,
        thickness,
        cam_index,
    )


def draw_points_in_image_color(
    points_uvd, image, max_distance=60, radius=1, thickness=-1
):
    """'
    Args:
        points_uvd: TensorShape(n, 3)
    """
    import matplotlib.pyplot as plt

    mask = (points_uvd[:, 0] > 0) & (points_uvd[:, 1] > 0) & (points_uvd[:, 2] > 0)
    points_uvd = points_uvd[mask]
    img = image.copy()
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(points_uvd.shape[0]):
        depth = points_uvd[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(
                np.round(points_uvd[i, 1]).astype(np.int),
                np.round(points_uvd[i, 0]).astype(np.int),
            ),
            radius=radius,
            color=tuple(color),
            thickness=thickness,
        )
    img = img.astype(np.uint8)
    return img


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1) + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, label, bev_size, bev_range=115, color=None, thickness=4
):
    color_flag = color is None
    id_class_map = get_id_class_map()
    colormap = get_task_colormap()
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
        lineType=cv2.LINE_AA,
        thickness=thickness,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
        lineType=cv2.LINE_AA,
        thickness=thickness,
    )
    if len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][..., [0, 1]]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if color_flag:
                    color = colormap[id_class_map[label[obj_idx]]]
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    else:
        vis_imgs = np.concatenate(
            [
                np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
            ],
            axis=0,
        )

    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    return vis_imgs


def draw_lidar_bbox3d_metas(
    bboxes_3d: torch.tensor,
    label: torch.tensor,
    imgs: torch.tensor,
    lidar2imgs: torch.tensor,
    imgs_meat: Dict,
    gt_depth: Union[List[torch.tensor], torch.tensor] = None,
    show_lidarpts=False,
):
    """imgs:RGB"""
    vis_imgs = []

    if show_lidarpts:
        bev = None

        ## method1: vis downsample lidarpoints
        # gt_depth = gt_depth.detach().cpu().numpy()  # (6,64,176)
        # N, H, W = gt_depth.shape
        # lidar_pts_img = np.ones([16 * N * H * W, 3]) * -1
        # grid_x1, grid_y1 = np.meshgrid(np.arange(N), np.arange(H * W))
        # grid_n = grid_x1.transpose(1, 0).flatten()  # (N*H*W,)
        # grid_x_ori, grid_y_ori = np.meshgrid(np.arange(W), np.arange(H))
        # grid_x = np.concatenate(N * [4 * grid_x_ori]).flatten()  # (N*H*W,) range(0, 4W)
        # grid_y = np.concatenate(N * [4 * grid_y_ori]).flatten()  # (N*H*W,) range(0, 4h)

        # flatten_idx = 16 * grid_n * H * W + 4 * grid_y * W + grid_x
        # lidar_pts_img[flatten_idx, 0] = grid_y
        # lidar_pts_img[flatten_idx, 1] = grid_x
        # depth = gt_depth.flatten()  # N*H*W
        # lidar_pts_img[flatten_idx, 2] = depth
        # lidar_pts_img = lidar_pts_img.reshape(N, 16 * H * W, 3)
        # for i, (img, lidar_pts) in enumerate(zip(imgs, lidar_pts_img)):
        #     img_show = draw_points_in_image_color(lidar_pts, img, thickness=1)
        #     vis_imgs.append(img_show)

        ## method2
        for _, (img, gtdepth) in enumerate(zip(imgs, gt_depth)):
            index = torch.nonzero(gtdepth)
            mask = gtdepth != 0
            depth = gtdepth[mask].reshape(-1, 1)
            lidar_pts_img = torch.concat([index, depth], 1)
            img_show = draw_points_in_image_color(
                lidar_pts_img.detach().cpu().numpy(), img, thickness=1
            )
            vis_imgs.append(img_show)

        num_imgs = len(vis_imgs)
        vis_imgs = np.concatenate(
            [
                np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
            ],
            axis=0,
        )
    else:
        for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
            img_show = draw_lidar_bbox3d_on_img(
                bboxes_3d, label, img, lidar2img, imgs_meat, cam_index=i
            )
            vis_imgs.append(img_show)

        vis_imgs = [
            vis_imgs[2],
            vis_imgs[0],
            vis_imgs[1],
            vis_imgs[5],
            vis_imgs[3],
            vis_imgs[4],
        ]
        num_imgs = len(vis_imgs)
        vis_imgs = np.concatenate(
            [
                np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
            ],
            axis=0,
        )
        bev = draw_lidar_bbox3d_on_bev(
            bboxes_3d, label.detach().cpu().numpy(), vis_imgs.shape[0], thickness=1
        )

    vis_imgs = np.concatenate([bev, vis_imgs], axis=1) if bev is not None else vis_imgs
    return vis_imgs


def video(imgs, dst_path, size):
    print("ori image size = ", size)
    size = (int(size[0] / 1), int(size[1] / 1))
    print("resized image size = ", size)
    resized_imgs = []
    for img in imgs:
        img = cv2.resize(img, dsize=size)
        resized_imgs.append(img)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst_name = dst_path +"sparse_end2end.mp4"
    videowrite = cv2.VideoWriter(dst_name, fourcc, 10, size)

    for i in tqdm(range(len(resized_imgs))):
        videowrite.write(resized_imgs[i])
    videowrite.release()
    print(f"Save video {dst_name} !")