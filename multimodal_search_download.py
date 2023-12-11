from pathlib import Path
import requests
import wikipedia
import urllib.request
import urllib.error
from PIL import Image
from io import BytesIO
import time

wiki_titles = [
    "batman",
    "Vincent van Gogh",
    "San Francisco",
    "iPhone",
    "Tesla Model S",
    "BTS",
]

data_dir = Path("data/multimodal_search")
docs_dir = data_dir / "docs"
docs_dir.mkdir(parents=True, exist_ok=True)
image_dir = data_dir / "images"
image_dir.mkdir(exist_ok=True)
image_uuid = 0
image_metadata_dict = {}
MAX_SIZE = 1024
MAX_IMAGES_PER_WIKI = 30

def open_url_with_retry(url, max_retries=10, delay=1):
    for retry in range(max_retries):
        try:
            # URLを開く
            response = urllib.request.urlopen(url)
            return response
        except urllib.error.URLError as e:
            print(f"Error: {e}")
            print(f"Retrying ({retry + 1}/{max_retries}) in {delay} seconds...")
            time.sleep(delay)

for title in wiki_titles:
    print(title)
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    with open(docs_dir / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

    if title == "BTS":
        title = "BTS band"
    images_per_wiki = 0
    page_py = wikipedia.page(title)
    list_img_urls = page_py.images
    for url in list_img_urls:
        if url.endswith(".jpg") or url.endswith(".png"):
            image_uuid += 1
            response = open_url_with_retry(url)
            if response is None:
                print(f"Failed to open {url}")
                continue
            data = response.read()
            image = Image.open(BytesIO(data))
            width, height = image.size
            if width > MAX_SIZE or height > MAX_SIZE:
                ratio = MAX_SIZE / max(width, height)
                image = image.resize((int(width * ratio), int(height * ratio)))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_dir / f"{image_uuid:03d}.jpg")
            
            images_per_wiki += 1
            if images_per_wiki > MAX_IMAGES_PER_WIKI:
                break