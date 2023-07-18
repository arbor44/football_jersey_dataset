# Football Jerseys Dataset

The repo contains all stuff to collect and process data for the project "This Football Jersey Doesn't Exist"

## Getting started

To start run:

```angular2html
pip install -r requirements.txt
```

## Data Downloading

### Images scraping

To scrap images from https://www.footballkitarchive.com run:

```angular2html
python data_scraping.py download-images ~/all_jerseys_images_folder
```
`all_jerseys_images_folder` -- the folder where the images will be stored. The structure of the folder
will be as follows:

```angular2html
images_dir
|---- league_1
      |---- team_1-season_1-kit_type.jpg
      |---- team_2-season_1-kit_type.jpg
      ...
|---- league_2
...
```
Here `team_1-season_1-kit_type.jpg`:

- `team_1` -- the name of the team (e.g. `liverpool`)
- `season_1` -- season (e.g. `1990-91`)
- `kit_type` -- the type of a kit (e.g. `home`, `away` etc.)

### Images filtering

Unfortunately, some images have lousy quality, and some images have problems with the background removal step (see later).
That's why you may need to filter datasets and separate these types of images. To do it run:

```angular2html
python data_scraping.py filter-images ~/all_jerseys_images_folder ~/jerseys_images_folder ./images_info/relevant_images.json
```

This command will save images from file `img_info/relevant_images.json` (you can find this file in this repo) in `~/jerseys_images_folder`;
the rest images will stay at `~/all_jerseys_images_folder`. https://www.footballkitarchive.com updates kits for new seasons;
you may want to pick some new jerseys with good quality that are not in `img_info/relevant_images.json`.

## Images preparation

To start training GAN, you need to prepare images for it: centre jerseys (jerseys are not centred by default) and remove
background (a lot of images have different backgrounds; first experiments without background removing have shown bad quality results;
after standardizing the background the quality of images has got much better).

### Jersey localisation
To localize a jersey on an image, I've trained the detection model (weights you can find [here](https://drive.google.com/file/d/1XNmAbycMGFlDlES_TGg57_at1QJUrfrQ/view?usp=sharing)) via [super-gradients](https://github.com/Deci-AI/super-gradients)
library and [deepfashion2](https://github.com/switchablenorms/DeepFashion2) dataset. From the dataset, I've extracted
images and bounding box annotations with classes `short_sleeve_top`, `long_sleeve_top`, `short_sleeve_outwear`,
`long_sleeve_outwear`, `short_sleeve_dress` and `long_sleeve_dress`; then I merged these classes in one class called
`top` and fined tune pre-trained [YOLO-NAS](https://deci.ai/blog/yolo-nas-object-detection-foundation-model/) model for my task.

To predict bounding boxes for jerseys run:
```angular2html
python jersey_preparation.py save-detection-results ~/yolo_nas_top_detection_model.pth ~/jerseys_images_folder ~/jerseys_images_folder/detection_results_dir
```

The command will predict bboxes and save to the folder `jerseys_images_folder/detection_results_dir` three files:
`one_box_predictions.json` (images with one bounding box), `more_one_box_predictions.json` (images with more than one bounding box),
`no_box_predictions.json` (images without any predictions). You can additionally annotate images from files `more_one_box_predictions.json`
and `no_box_predictions.json`. Images in `one_box_predictions.json` have one box, and you can use these detection results
(as is) to crop jerseys.

But there can be issues with bbox localization on some images. If you want to use well-localised bboxes for filtered
images (see [Images filtering part](#images-filtering) in Data Downloading chapter), you can use file `images_info/jerseys_bounding_boxes.json`. I've manually fixed the bad
localised bboxes for images in this file.

### Background removal

As I mentioned above, the important step of data processing is background removal. Experiments with white background have
shown a much better quality of generated images. There are a lot of jerseys on the white background in the initial dataset.
The images for which the background needs to be removed, are stored in the file `images_info/images_to_remove_background.json`.
To make the background on these images white, run:

```angular2html
python jersey_preparation.py remove-background ~/jerseys_images_folder ./images_info/images_to_remove_background.json
```

If you want to save images with initial background to the separate folder, run:

```angular2html
python jersey_preparation.py remove-background ~/jerseys_images_folder ./images_info/images_to_remove_background.json \\
    --save_original --path_to_save_original ~/original_images_with_background_folder
```

### Images standardisation

To standardise images (crop to jerse's box, resize and pad to one shape), run:

```angular2html
python jersey_preparation.py prepare-images ~/jerseys_images_folder ./images_info/jerseys_bounding_boxes.json \\
    ~/prepared_images --size 512
```

Finally, you have the dataset ready for GAN training!

