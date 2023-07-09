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
    Get detection results for supergradients detection pipeline

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
    net = models.get(f"yolo_nas_{yolo_nas_size}", num_classes=1, checkpoint_path=yolo_nas_model_path)
    default_nms = net._default_nms_conf

    if default_nms < lowest_nms_conf:
        raise ValueError(f"The lowest non max suppression confidence threshold (equal to {lowest_nms_conf}) should be"
                         f" lower than default non max suppression confidence (equal to {default_nms})")

    one_box_predictions, more_one_box_predictions = {}, {}

    images_paths = list(map(str, Path(input_images_dir).rglob("*.jpg")))

    for nms_conf in np.arange(default_nms, lowest_nms_conf, -.05):
        pipeline = net._get_pipeline(conf=nms_conf)
        start, stop = 0, batch_size

        images_to_predict = list(set(images_paths) - set(one_box_predictions) - set(more_one_box_predictions))

        logger.info(f"Jersey detections for NMS confidence threshold {nms_conf}")
        pbar = tqdm(total=len(images_to_predict))
        while stop < len(images_to_predict) + batch_size:
            images = list(map(lambda x: cv.imread(x)[..., ::-1], images_to_predict[start:stop]))
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
@click.argument("path_to_json_with_images")
@click.argument("output_folder")
@click.option("--n_jobs", default=5, type=int)
def remove_background(path_to_json_with_images: str, output_folder: str, n_jobs: int):
    path_to_images = load_json(path_to_json_with_images)
    output_folder = Path(output_folder)

    def make_removal(img_path: Path):
        out_dir = output_folder / img_path.parents[0].name
        img_rgba = remove(cv.imread(str(img_path))[..., ::-1])

        alpha = img_rgba[..., 3]
        alpha3 = np.dstack((alpha, alpha, alpha))
        img_black_background = (img_rgba[..., 0:3].astype(np.float32) * (alpha3 / 255.)).astype(np.uint8)
        img_white_background = np.clip(img_black_background + (255 - alpha3), 0, 255)

        cv.imwrite(str(out_dir / img_path.name), img_white_background[..., ::-1])

    logger.info("Create leagues' dirs")
    for img_path in tqdm(list(map(Path, path_to_images))):
        out_dir = output_folder / img_path.parents[0].name
        if not out_dir.exists(): out_dir.mkdir(exist_ok=True)

    logger.info("Remove background")
    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(make_removal)(img_path) for img_path in tqdm(list(map(Path, path_to_images))))


@main.command()
@click.argument("path_to_detection_results")
@click.argument("output_dir")
@click.option("--size", default=256, type=int)
@click.option("--n_jobs", default=20, type=int)
def prepare_images(path_to_detection_results: str, output_dir: str, size: int, n_jobs: int):
    detection_results = load_json(path_to_detection_results)
    output_dir = Path(output_dir)

    if not output_dir.exists(): output_dir.mkdir()

    def save_image(img_path: str, box: List[float]):
        img = cv.imread(img_path)[..., ::-1]
        processed_image = get_prepared_image(img, box, size, (255, 255, 255))

        cv.imwrite(str(output_dir / Path(img_path).name), processed_image[..., ::-1])

    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(save_image)(img_path, vals["boxes"][0]) for img_path, vals in tqdm(detection_results.items()))


if __name__ == "__main__":
    main()
