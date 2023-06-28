import logging
import click

from pathlib import Path

from dataprep.scraping import get_and_save_all_images_by_league
from dataprep.utils import save_json, configure_logger

logger = configure_logger(logging.getLogger(__name__), 2)

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

