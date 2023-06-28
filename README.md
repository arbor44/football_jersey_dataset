# Football Jerseys Dataset

The repo contains all stuff to collect and process data for the project "This Football Jersey Doesn't Exist"

## Getting started

To start run:

```angular2html
pip install -r requirements.txt
```

## Data Scraping

To scrap images from https://www.footballkitarchive.com run:

```angular2html
python data_scraping.py images_dir --n_jobs desired_num_of_jobs
```
`images_dir` -- the folder where the images will be stored. The structure of the folder will be as follows:

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
