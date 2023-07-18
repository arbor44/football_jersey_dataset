import shutil

import click
import logging
import torch

import numpy as np
import cv2 as cv

from joblib import Parallel, delayed
from pathlib import Path
from rembg import remove
from tqdm import tqdm
from typing import List, Tuple, Union

from super_gradients.training import models
from super_gradients.training.pipelines.pipelines import DetectionPipeline, eval_mode
from super_gradients.training.models.predictions import DetectionPrediction


from dataprep.utils import save_json, load_json, configure_logger
from dataprep.cv_utils.spatial_operations.spatial_operations import crop_image_to_xyxy_box, pad_to_shape, resize_image
from dataprep.cv_utils.spatial_operations.params_extractors import get_adjusted_to_image_borders_coords


def get_yolo_nas_predictions(pipeline: DetectionPipeline, images: List[np.ndarray]) -> List[DetectionPrediction]:
    """
    Get detection results for supergradients detection pipeline. The function is refactored variant of
    https://github.com/Deci-AI/super-gradients/blob/18b02f007adb1a18c5935ccf476e5fcfcd93bc2d/src/super_gradients/training/pipelines/pipelines.py#L160

    :param pipeline: supergradients detection pipeline
    :param images: images to which detection pipeline will be applied

    :return: list of processed detection predictions
    """
    # Make sure the model is on the correct device, as it might have been moved after init
    pipeline.model = pipeline.model.to(pipeline.device)

    # Preprocess
    preprocessed_images, processing_metadatas = [], []
    for image in images:
        preprocessed_image, processing_metadata = pipeline.image_processor.preprocess_image(image=image.copy())
        preprocessed_images.append(preprocessed_image)
        processing_metadatas.append(processing_metadata)

    # Predict
    with eval_mode(pipeline.model), torch.no_grad(), torch.cuda.amp.autocast():
        torch_inputs = torch.from_numpy(np.array(preprocessed_images)).to(pipeline.device)
        model_output = pipeline.model(torch_inputs)
        predictions = pipeline._decode_model_output(model_output, model_input=torch_inputs)

    # Postprocess
    postprocessed_predictions = []
    for image, prediction, processing_metadata in zip(images, predictions, processing_metadatas):
        prediction = pipeline.image_processor.postprocess_predictions(predictions=prediction,
                                                                      metadata=processing_metadata)
        postprocessed_predictions.append(prediction)

    return postprocessed_predictions


def process_detection_results(detection_res: DetectionPrediction) -> List[List[float]]:
    """
    Convert supergradients DetectionPrediction to box in format
    (x_upper_left_corner, y_upper_left_corner, x_bottom_right_corner, y_bottom_right_corner, confidence)

    :param detection_res: supergradients model detection results

    :return: boxes coords with confidence
    """
    boxes, confidences = detection_res.bboxes_xyxy, detection_res.confidence
    if len(boxes) == 0: return []

    return np.hstack([boxes, confidences.reshape(-1, 1)]).tolist()


def get_prepared_image(
        image: np.ndarray, box: np.ndarray, size: int, padding_value: Union[Tuple[int, int, int], int]) -> np.ndarray:
    """
    Process image (crop image to box, rescale with same aspect ratio and pad to size x size shape)

    :param image: image to process
    :param box: bounding box to crop image with
    :param size: processed image target size
    :param padding_value: value to pad with

    :return: processed image
    """
    height, width = image.shape[:2]
    box = get_adjusted_to_image_borders_coords(np.array(box[:4]).reshape(2, 2), width, height).flatten()

    processed_image = pad_to_shape(
        resize_image(crop_image_to_xyxy_box(image, box), size, size), size, size, padding_value=padding_value)

    return processed_image
# ----------------------------------------------------------------------------


logger = configure_logger(logging.getLogger(__name__), 2)


# ----------------------------------------------------------------------------


@click.group()
def main():
    """
    Tools set to prepare jerseys' images for GAN training.

    Example:

    \b
    # Remove background from images in images_to_remove_background.json and make it white. Original images will be
    # rewritten by new images
    python jersey_preparation.py remove-background ~/jerseys_images_folder ./images_info/images_to_remove_background.json

    \b
    # Remove background from images in images_to_remove_background.json and make it white. Original images will be
    # saved in original_images_with_background_folder
    python jersey_preparation.py remove-background ~/jerseys_images_folder ./images_info/images_to_remove_background.json \\
        --save_original --path_to_save_original ~/original_images_with_background_folder

    \b
    # Prepare jerseys' images from ~/jerseys_images_folder for training (crop to jerse's box, resize and pad to shape
    # 512x512) and save them in folder ~/prepared_images
    python jersey_preparation.py prepare-images ~/jerseys_images_folder ./images_info/jerseys_bounding_boxes.json \\
        ~/prepared_images --size 512

    \b
    # Get bounding boxes for jerseys in the folder ~/jerseys_images_folder via model stored in yolo_nas_model.pth \\
    # and save detection results to the folder ~/jerseys_images_folder/detection_results_dir
    python jersey_preparation.py save-detection-results ~/yolo_nas_model.pth ~/jerseys_images_folder ~/jerseys_images_folder/detection_results_dir
    """
    pass


@main.command()
@click.argument("yolo_nas_model_path")
@click.argument("input_images_dir")
@click.argument("results_dir")
@click.option("--yolo_nas_size", default="s", type=str)
@click.option("--batch_size", default=32, type=int)
@click.option("--lowest_nms_conf", default=.1, type=float,)
def save_detection_results(
    yolo_nas_model_path: str,
    input_images_dir: str,
    results_dir: str,
    yolo_nas_size: str,
    batch_size: int,
    lowest_nms_conf: float
):
    """
    Detect jerseys on images and save detection results; detection will start with the default non-max-suppression (NMS)
    threshold and will run again on images with empty predictions with lower NMS threshold until lowest_nms_conf will
    be reached; 3 JSON file will be saved to results_dir: one_box_predictions.json (images with one bounding box),
    more_one_box_predictions.json (images with more than one bounding box), no_box_predictions.json
    (images without any predictions)

    :param yolo_nas_model_path: path to yolo nas model
    :param input_images_dir: folder with images to process
    :param results_dir: folder to save results
    :param yolo_nas_size: size of yolo nas model (e.g. "s", "m", "l")
    :param batch_size: batch size for yolo nas model
    :param lowest_nms_conf: the lowest non-max-suppression threshold to check
    """
    net = models.get(f"yolo_nas_{yolo_nas_size}", num_classes=1, checkpoint_path=yolo_nas_model_path)
    default_nms = net._default_nms_conf

    if default_nms < lowest_nms_conf:
        raise ValueError(f"The lowest non max suppression confidence threshold (equal to {lowest_nms_conf}) should be"
                         f" lower than default non max suppression confidence (equal to {default_nms})")

    one_box_predictions, more_one_box_predictions = {}, {}

    images_paths = list(map(lambda x: str(x.relative_to(input_images_dir)), Path(input_images_dir).rglob("*.jpg")))

    for nms_conf in np.arange(default_nms, lowest_nms_conf, -.05):
        pipeline = net._get_pipeline(conf=nms_conf)
        start, stop = 0, batch_size

        images_to_predict = list(set(images_paths) - set(one_box_predictions) - set(more_one_box_predictions))

        logger.info(f"Jersey detections for NMS confidence threshold {nms_conf}")
        pbar = tqdm(total=len(images_to_predict))
        while stop < len(images_to_predict) + batch_size:
            images = list(
                map(lambda x: cv.imread(str(Path(input_images_dir) / x))[..., ::-1], images_to_predict[start:stop]))
            detection_results = list(map(process_detection_results, get_yolo_nas_predictions(pipeline, images)))

            for i, img_name in enumerate(images_to_predict[start:stop]):
                if len(detection_results[i]) == 0:
                    continue

                if len(detection_results[i]) > 1:
                    more_one_box_predictions[img_name] = {"boxes": detection_results[i], "shape": images[i].shape[:2]}
                    continue

                if len(detection_results[i]) == 1:
                    one_box_predictions[img_name] = {"boxes": detection_results[i], "shape": images[i].shape[:2]}

            start, stop = stop, stop + batch_size
            pbar.update(stop - start)

        pbar.close()

    save_json(one_box_predictions, Path(results_dir) / "one_box_predictions.json")
    save_json(more_one_box_predictions, Path(results_dir) / "more_one_box_predictions.json")
    save_json(
        list(set(images_paths) - set(one_box_predictions) - set(more_one_box_predictions)),
        Path(results_dir) / "no_box_predictions.json")


@main.command()
@click.argument("input_images_dir")
@click.argument("path_to_json_with_images")
@click.option("--n_jobs", default=5, type=int)
@click.option("--save_original", is_flag=True, default=False)
@click.option("--path_to_save_original", default=None)
def remove_background(
    input_images_dir: str,
    path_to_json_with_images: str,
    n_jobs: int,
    save_original: bool,
    path_to_save_original: str
):
    """
    Remove background for images in path_to_json_with_images and save images with white background instead original
    images; it supposes that paths inside path_to_json_with_images have the structure league_name/image_name.jpg and
    images are located in input_images_dir (full path to the image is input_images_dir/league_name/image_name.jpg)

    in case save_original is True, original images will be saved as path_to_save_original/league_name/image_name.jpg

    :param input_images_dir: folder with images to process
    :param path_to_json_with_images: path to JSON file with the list of relative to input_images_dir paths to images
                                     to remove the background
    :param n_jobs: number of jobs to use in parallel
    :param save_original: save or not original images; default False
    :param path_to_save_original: folder to save original images in case it is necessary
                                  (will be used only if save_original is True)

    """
    relative_paths_to_images = load_json(path_to_json_with_images)
    input_images_dir = Path(input_images_dir)

    if save_original:
        if path_to_save_original is None:
            raise ValueError("There must be path_to_save_original if save_original is True.")

        path_to_save_original = Path(path_to_save_original)

        logger.info("Create leagues' dirs to save original images")
        for img_path in tqdm(list(map(Path, relative_paths_to_images))):
            out_dir = path_to_save_original / img_path.parents[0].name
            out_dir.mkdir(exist_ok=True, parents=True)

    def make_removal(relative_img_path: Path):
        """
        Background removal to execute in parallel
        """
        img_path = input_images_dir / relative_img_path

        if not img_path.exists():
            logger.info(f"The image {relative_img_path} was not downloaded from https://www.footballkitarchive.com. "
                        f"It can be because of some changes on https://www.footballkitarchive.com")
            return

        img_rgba = remove(cv.imread(str(img_path))[..., ::-1])

        alpha = img_rgba[..., 3]
        alpha3 = np.dstack((alpha, alpha, alpha))
        img_black_background = (img_rgba[..., 0:3].astype(np.float32) * (alpha3 / 255.)).astype(np.uint8)
        img_white_background = np.clip(img_black_background + (255 - alpha3), 0, 255)

        if save_original:
            shutil.move(img_path, path_to_save_original / relative_img_path)

        cv.imwrite(str(img_path), img_white_background[..., ::-1])

    logger.info("Remove background")
    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(make_removal)(img_path) for img_path in tqdm(list(map(Path, relative_paths_to_images))))


@main.command()
@click.argument("input_images_dir")
@click.argument("path_to_detection_results")
@click.argument("output_dir")
@click.option("--size", default=256, type=int)
@click.option("--n_jobs", default=20, type=int)
def prepare_images(input_images_dir: str, path_to_detection_results: str, output_dir: str, size: int, n_jobs: int):
    """
    Prepare images for GAN training: crop image to jersey's box, rescale and pad to size x size shape
    (padding with white background); it supposes that paths inside path_to_detection_results have the structure
    league_name/image_name.jpg and images are located in input_images_dir (full path to the image is
    input_images_dir/league_name/image_name.jpg); images will be saved in output_folder with name image_name.jpg

    :param input_images_dir: folder with images to process
    :param path_to_detection_results: JSON file with detection results; the file inside is a dict
                                      {"league_name/image_name.jpg": bboxes} (the structure is the same as in
                                      one_box_predictions.json from save_detection_results function)
    :param output_dir: folder to save prepared images
    :param size: final size of images
    :param n_jobs: number of jobs to use in parallel
    """
    input_images_dir = Path(input_images_dir)
    detection_results = load_json(path_to_detection_results)
    output_dir = Path(output_dir)

    if not output_dir.exists(): output_dir.mkdir()

    def save_image(img_path: str, box: List[float]):
        if not (input_images_dir / img_path).exists():
            logger.info(f"The image {img_path} was not downloaded from https://www.footballkitarchive.com. "
                        f"It can be because of some changes on https://www.footballkitarchive.com")
            return

        img = cv.imread(str(input_images_dir / img_path))[..., ::-1]
        processed_image = get_prepared_image(img, box, size, (255, 255, 255))

        cv.imwrite(str(output_dir / Path(img_path).name), processed_image[..., ::-1])

    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(save_image)(img_path, vals["boxes"][0]) for img_path, vals in tqdm(detection_results.items()))

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
