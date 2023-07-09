import numpy as np
from typing import Sequence, Tuple, Union


def convert_xyxy_box_to_upper_left_wh_box(
        xyxy_box: Sequence[Union[int, float]]
) -> Tuple[Union[int, float],  Union[int, float], Union[int, float], Union[int, float]]:
    """
    Convert bbox from format [x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner]
    to format ((x_upper_left_corner, y_upper_left_corner), width, height)

    :param xyxy_box: [x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner]
    :returns: (x_upper_left_corner, y_upper_left_corner, width, height)
    """
    width, height = xyxy_box[2] - xyxy_box[0], xyxy_box[3] - xyxy_box[1]
    return xyxy_box[0], xyxy_box[1], width, height


def convert_xywh_box_to_xyxy_box(
        xywh_box: Sequence[Union[int, float]]
) -> Tuple[Union[int, float],  Union[int, float], Union[int, float], Union[int, float]]:
    """
    Convert box in format (x_center, y_center, width, height) to format
    (x_upper_left, y_upper_left, x_bottom_right, y_bottom_right)

    :param xywh_box: box in format (x_center, y_center, width, height)
    :return: box in format (x_upper_left, y_upper_left, x_bottom_right, y_bottom_right)
    """
    x_center, y_center, w, h = xywh_box
    x_upper_left, x_bottom_right = np.array([-w / 2, w / 2]) + x_center
    y_upper_left, y_bottom_right = np.array([-h / 2, h / 2]) + y_center

    return x_upper_left, y_upper_left, x_bottom_right, y_bottom_right


def convert_xyxy_box_to_relative_box(
        xyxy_box: Sequence[Union[int, float]], shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Convert bbox from format (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)
    to format (x_upper_left_relative, y_upper_left_relative, width_relative, height_relative)

    :param xyxy_box: box in format
                    (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)
    :param shape: shape of image in format (H, W)
    :return: box in format (x_upper_left_relative, y_upper_left_relative, width_relative, height_relative)
    """
    x1, y1, x2, y2 = xyxy_box
    h, w = shape
    rel_box = (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)

    return rel_box


def convert_relative_box_to_xyxy_box(
        relative_box: Sequence[float], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Convert bbox from format (x_upper_left_relative, y_upper_left_relative, width_relative, height_relative)
    to format (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)

    :param relative_box: box in format (x_upper_left_relative, y_upper_left_relative, width_relative, height_relative)
    :param shape: shape of image in format (H, W)
    :return: box in format (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)
    """
    x_rel, y_rel, w_rel, h_rel = relative_box
    h, w = shape
    xyxy_box = np.int32([x_rel * w, y_rel * h, (x_rel + w_rel) * w, (y_rel + h_rel) * h])

    return tuple(xyxy_box)


def convert_xysr_box_to_xywh_box(xysr_box: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Convert bbox from format (x_center, y_center, width * height, weight / height)
    to format (x_center, y_center, width, height)

    :param xysr_box: bbox in format (x_center, y_center, width * height, weight / height)
    :return: bbox in format (x_center, y_center, width, height)
    """
    x, y, s, r = xysr_box
    return x, y, np.sqrt(s * r), np.sqrt(s / r)


def convert_xywh_box_to_xysr_box(xywh_box: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Convert bbox from format (x_center, y_center, width, height)
    to format (x_center, y_center, width * height, weight / height)

    :param xywh_box: bbox in format (x_center, y_center, width, height)
    :return: bbox in format (x_center, y_center, width * height, weight / height)
    """

    x, y, w, h = xywh_box
    return x, y, w * h, w / h


def convert_xyxy_box_to_xysr_box(xyxy_box: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Convert bbox from format (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)
    to format (x_center, y_center, width * height, weight / height)

    :param xyxy_box: bbox in format
                    (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner)
    :return: bbox in format (x_center, y_center, width * height, weight / height)
    """
    x_ul, y_ul, x_br, y_br, *_ = xyxy_box
    w, h = x_br - x_ul, y_br - y_ul
    x_center, y_center = x_ul + w / 2., y_ul + h / 2.
    s, r = w * h, w / float(h+1e-6)

    return x_center, y_center, s, r
