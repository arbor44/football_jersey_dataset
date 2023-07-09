import numpy as np
import cv2 as cv

from typing import Tuple, Union

from .params_extractors import get_rescaled_size, get_pad_params


def resize_image(image: np.ndarray,
                 target_height: int,
                 target_width: int,
                 save_scale_ratio: bool = True,
                 interpolation: int = cv.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to target size (could change in case we want save width to height ratio)

    :param image: image to resize
    :param target_height: desired height
    :param target_width: desired width
    :param save_scale_ratio: save or not width to height ratio; in case of True target size could change
    :param interpolation: opencv interpolation flag

    :return: resized image
    """

    resized_h, resized_w = get_rescaled_size(target_height, target_width, *image.shape[:2], save_scale_ratio)

    if image.shape[:2] == (resized_h, resized_w): return image

    return cv.resize(image, dsize=(resized_w, resized_h), interpolation=interpolation)


def pad_to_shape(image: np.ndarray,
                 target_height: int,
                 target_width: int,
                 border_mode: int = cv.BORDER_CONSTANT,
                 padding_value: Union[Tuple[int, int, int], int] = 0) -> np.ndarray:
    """
    Pad image to desired size

    :param image: image to pad
    :param target_height: desired height
    :param target_width: desired width
    :param border_mode: opencv border mode flag
    :param padding_value: value to insert in padding area

    :return: padded image
    """

    h_pad_top, h_pad_down, w_pad_left, w_pad_right = get_pad_params(target_height, target_width, *image.shape[:2])

    if h_pad_top + h_pad_down + w_pad_right + w_pad_left == 0: return image

    return cv.copyMakeBorder(image, top=h_pad_top, bottom=h_pad_down, left=w_pad_left, right=w_pad_right,
                             borderType=border_mode, value=padding_value)


def crop_image_to_xyxy_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Crop image to box in format [x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner]

    :param image: image array
    :param box: bbox in format [x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner]

    :return: cropped image
    """
    x_tl, y_tl, x_br, y_br = np.int64(box[:4])
    return image[y_tl:y_br, x_tl:x_br, ...]