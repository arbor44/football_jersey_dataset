"""
Stuff to scrap the data from the website https://www.footballkitarchive.com
"""

import click

import numpy as np
import cv2 as cv
import logging

import bs4
import requests
import xmltodict

from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union

from dataprep.utils import save_json, configure_logger


def image_loader(image_url: str, num_elements: int = 1000) -> Tuple[Union[np.ndarray, None], dict]:
    """
    Load image via url

    :param image_url: url of the image to download
    :param num_elements: min number of elements should be in the image; in case th number of elements less, return None

    :return: image and response info
    """
    image_request = requests.get(image_url)

    # save info about the last request
    info = {"ok": image_request.ok, "status": image_request.status_code}
    if not image_request.ok: return None, info

    # bytes to array
    image = np.frombuffer(image_request.content, np.uint8)

    # check are there some bullshit or not
    info.update({"content_num_elements": len(image)})
    if len(image) <= num_elements: return None, info

    # try to decode image
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image, info


def save_kit_for_teams(team: bs4.Tag, path_to_league: Path, logger: logging.Logger) -> List[Tuple[str, str]]:
    """
    Find all kits for team and download it

    :param team: the info about the team inside liga's season webpage
                 (like this https://www.footballkitarchive.com/bundesliga-2023-24-kits/)
    :param path_to_league: local path to folder with kits for league
    :param logger: logger to log errors

    :return: list with info (kit source, kit name) about saved kits
    """
    kits = []
    for kit in team.findAll("div", class_="kit"):
        kit_name = kit.text.replace(" ", "-").replace("\n", "-").replace(".", "").strip("-").lower()
        kit_source = kit.find("img")["src"].split("-small")[0] + ".jpg"

        try:
            image, _ = image_loader(kit_source)
        except:
            logger.error(f'Problems with kit {kit_name} (high resolution source: {kit_source},'
                         f' low resolution source: {kit.find("img")["src"]}')
            image = None

        if image is not None:
            cv.imwrite(str(path_to_league / (kit_name + ".jpg")), image)
            kits.append((kit_source, kit_name))

    return kits


def get_and_save_all_images_by_league(
        league_name: str, path_to_images: Path, n_jobs: int, logger: logging.Logger) -> List[Tuple[str, str]]:
    """
    Save all images for the league on the website https://www.footballkitarchive.com in parallel

    :param league_name: the name of league on the website https://www.footballkitarchive.com
    :param path_to_images: local path to save images
    :param n_jobs: number of jobs to use in parallel for images downloading
    :param logger: logger to log info

    :return: list with info (kit source, kit name) about saved kits
    """
    url = "https://www.footballkitarchive.com"

    logger.info(f"Save images for {league_name} league")

    path_to_league = path_to_images / league_name
    path_to_league.mkdir(exist_ok=True)

    league_page = requests.get(url + f"/{league_name}-kits/")
    league_soup = bs4.BeautifulSoup(league_page.content, "html.parser")

    seasons = list(map(
        lambda x: xmltodict.parse(str(x))["header"]['h3']['a']['@href'],
        league_soup.find_all("header", class_="collection-header")))

    kits = []
    for season in tqdm(seasons):
        season_page = requests.get(url + season)
        season_soup = bs4.BeautifulSoup(season_page.content, "html.parser")
        teams = season_soup.find_all("div", class_="collection-kits")

        kits.extend(Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(save_kit_for_teams)(team, path_to_league, logger) for team in teams))

    return kits


# ----------------------------------------------------------------------------

leagues = ["bundesliga", "premier-league", "division-1", "serie-a", "world-cup", "superliga-argentina",
           "la-liga", "ligue-1", "mls", "austrian-bundesliga", "swiss-super-league", "championship",
           "scottish-premiership", "eredivisie", "danish-superliga", "serbian-superliga", "estonian-meistriliiga",
           "cypriot-first-division", "super-league-greece", "super-lig", "brasileiro-serie-a", "ekstraklasa",
           "czech-first-league", "russian-premier-league", "ukrainian-premier-league", "slovak-super-liga",
           "nemzeti-bajnoksag", "allsvenskan", "eliteserien", "veikkausliiga", "faroe-islands-premier-league",
           "belgian-first-division-a", "belgian-first-division-b", "primera-division-de-uruguay",
           "chinese-super-league", "k-league", "canadian-premier-league", "primera-a-de-mexico", "slovenian-prvaliga",
           "romanian-liga-i", "saudi-pro-league", "paraguayan-primera-division", "liga-1-peru", "ecuadorian-serie-a",
           "categoria-primera-a", "premier-league-of-bosnia-and-herzegovina", "kazakhstan-premier-league",
           "indian-super-league", "soviet-top-league", "v-league-1", "thai-league-1", "asian-cup", "euro",
           "copa-america", "africa-cup", "south-african-national-first-division", "conifa", "primera-division-de-chile",
           "serie-b", "segunda-division-b", "ligue-2", "2-bundesliga", "brasileiro-serie-b",
           "brasileiro-serie-c", "eerste-divisie", "segunda-division-de-uruguay", "swiss-challenge-league",
           "2-liga-austria", "liga-portugal-2", "primera-nacional"]

logger = configure_logger(logging.getLogger(__name__), 2)


# ----------------------------------------------------------------------------

@click.command()
@click.argument("path_to_images")
@click.option("--n_jobs", default=20, type=int, help="the number of frames per chunk")
def download_images(path_to_images: str, n_jobs: int):
    path_to_images = Path(path_to_images)

    kits = []
    for league in leagues:
        kits.extend(get_and_save_all_images_by_league(league, path_to_images, n_jobs, logger))

    save_json(path_to_images / "saved_images.json")


if __name__ == "__main__":
    download_images()
