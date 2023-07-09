import numpy as np
from typing import Tuple


def get_rescaled_size(
        target_height: int, target_width: int, image_height: int, image_width: int, save_scale_ratio: bool = True
) -> Tuple[int, int]:
    """
    Get height and width to which adjust during rescaling

    :param target_height: desired height
    :param target_width: desired width
    :param image_height: current image height
    :param image_width: current image width
    :param save_scale_ratio: save or not width to height ratio; in case of True target size could change

    :return: height, width to which adjust
    """

    if not save_scale_ratio:
        return target_height, target_width

    if target_height == image_height and target_width == image_width:
        return target_height, target_width

    resized_h, resized_w = target_height, target_width

    if image_height * target_width > image_width * target_height:
        resized_w = int(image_width * target_height / image_height)
    elif image_height * target_width < image_width * target_height:
        resized_h = int(image_height * target_width / image_width)

    return resized_h, resized_w


def get_pad_params(
        target_height: int, target_width: int, image_height: int, image_width: int) -> Tuple[int, int, int, int]:
    """
    Get margins to add to image during padding procedure

    :param target_height: desired height
    :param target_width: desired width
    :param image_height: current image height
    :param image_width: current image width

    :return: top, down, left and right margins for padding
    """

    if image_height < target_height:
        h_pad_top = (target_height - image_height) // 2
        h_pad_down = target_height - image_height - h_pad_top
    else:
        h_pad_top, h_pad_down = 0, 0

    if image_width < target_width:
        w_pad_left = (target_width - image_width) // 2
        w_pad_right = target_width - image_width - w_pad_left
    else:
        w_pad_left, w_pad_right = 0, 0

    return h_pad_top, h_pad_down, w_pad_left, w_pad_right


def get_inverse_shift_scale(
        target_height: int, target_width: int, image_height: int, image_width: int, save_scale_ratio: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get scale, shift and image size to make inverse transform (unpad, unrescale and adjust to image size)

    :param target_height: desired height
    :param target_width: desired width
    :param image_height: current image height
    :param image_width: current image width
    :param save_scale_ratio: save or not width to height ratio during rescaling

    :return: scale, shift and image size (width, height) params for each axis
    """

    resized_h, resized_w = get_rescaled_size(target_height, target_width, image_height, image_width, save_scale_ratio)
    h_pad_top, _, w_pad_left, _ = get_pad_params(target_height, target_width, resized_h, resized_w)

    scale_x, scale_y = image_width / resized_w, image_height / resized_h
    shift_x, shift_y = -w_pad_left, -h_pad_top

    return np.array([scale_x, scale_y]), np.array([shift_x, shift_y]), np.array([image_width, image_height])


def get_adjusted_to_image_borders_coords(
        xy_coords: np.ndarray, image_width: int, image_height: int, margin: int = 0) -> np.ndarray:
    """
    Adjust points coordinates to image borders

    :param xy_coords: point coords in format np.array([[x, y], ...])
    :param image_width: width of image
    :param image_height: height of image
    :param margin: num pixels to adjust from image borders

    :return: adjusted coordinates
    """
    num_points = len(xy_coords)
    upper_left_border = np.array([margin, margin] * num_points).reshape(-1, 2)
    bottom_right_border = np.array([image_width - margin, image_height - margin] * num_points).reshape(-1, 2)

    return np.min([np.max([xy_coords, upper_left_border], axis=0), bottom_right_border], axis=0)
