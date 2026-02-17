# backend/download_images.py (updated)
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BASE_DIR = os.path.dirname(__file__)
OUTFITS = os.path.join(BASE_DIR, "outfits")
THUMBS = os.path.join(BASE_DIR, "thumbs")
os.makedirs(OUTFITS, exist_ok=True)
os.makedirs(THUMBS, exist_ok=True)

# real remote images (the 11 you gave)
to_download = {
    "upper00.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/00_upper.jpg",
    "upper01.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/01_upper.jpg",
    "upper02.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/02_upper.png",
    "upper03.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/03_upper.jpg",
    "dress04.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/04_dress.png",
    "dress05.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/05_dress.jpg",
    "upper06.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/06_upper.png",
    "upper07.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/07_upper.png",
    "upper08.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/08_upper.png",
    "upper09.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/09_upper.png",
    "dress10.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/10_dress.png",
    "upper11.png": "https://huggingface.co/spaces/mdshish61/Kolors-Virtual-Try-On/resolve/main/assets/cloth/11_upper.png",
}

# extra placeholder items we'll create locally (names -> display text)
placeholders = {
    # tops/uppers
    "upper12.png": "Red Floral Top",
    "upper13.png": "Black Crop Top",
    "upper14.png": "Green Hoodie",
    "upper15.png": "Yellow Kurti",
    "upper16.png": "Grey Sweater",
    "upper20.png": "Denim Jacket",
    # bottoms
    "pant02.png": "Blue Jeans",
    "pant03.png": "Trousers",
    "pant04.png": "Black Skirt",
    "pant05.png": "Denim Shorts",
    # dresses / ethnic
    "dress11.png": "Orange Dress",
    "saree01.png": "Blue Saree",
    "kurti01.png": "Kurti",
    # shoes & accessories
    "shoes01.png": "Brown Loafer",
    "shoes02.png": "White Sneakers",
    "shoes03.png": "Black Boots",
    "shoes04.png": "Running Shoes",
    "bag01.png": "Leather Bag",
    "sunglasses01.png": "Sunglasses",
    "cap01.png": "Cap",
    # watches
    "watch01.png": "Silver Watch",
    "watch02.png": "Gold Watch",
    "watch03.png": "Sport Watch",
    # extra tops
    "upper17.png": "Pink Top"
}

def download_and_save(name, url):
    print("Downloading", url)
    try:
        resp = urllib.request.urlopen(url, timeout=20)
        content = resp.read()
        img = Image.open(BytesIO(content)).convert("RGBA")
        save_path = os.path.join(OUTFITS, name)
        img.save(save_path)
        print("Saved", save_path)
        # create thumbnail
        thumb = img.copy()
        
        thumb.thumbnail((300,300), Image.LANCZOS)

        thumb.save(os.path.join(THUMBS, name))
        print("Thumb saved", os.path.join(THUMBS, name))
    except Exception as e:
        print("Failed to download", url, "->", e)

def make_placeholder(name, text, size=(800,1000), bg=None):
    # create a simple but clean placeholder PNG and thumbnail
    if bg is None:
        # choose background color by hashing name
        h = abs(hash(name)) % 200
        bg = (180 + h % 40, 160 + (h*3)%40, 200 - (h%30))
    img = Image.new("RGBA", size, bg + (255,))
    draw = ImageDraw.Draw(img)
    # large label in center
    try:
        fsize = 48
        # try to get a truetype font if available
        try:
            font = ImageFont.truetype("arial.ttf", fsize)
        except Exception:
            font = None
        w, hh = draw.textsize(text, font=font)
        x = (size[0] - w)//2
        y = (size[1] - hh)//2
        # outline text for readability
        outline_color = (0,0,0)
        if font:
            # draw thin outline
            for ox, oy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                draw.text((x+ox,y+oy), text, font=font, fill=outline_color)
        draw.text((x,y), text, font=font, fill=(255,255,255))
    except Exception:
        draw.text((20,20), text, fill=(0,0,0))
    # save full and thumb
    fullpath = os.path.join(OUTFITS, name)
    thumbpath = os.path.join(THUMBS, name)
    img.save(fullpath)
    thumb = img.copy()
    
    thumb.thumbnail((300,300), Image.LANCZOS)

    thumb.save(thumbpath)
    print("Created placeholder", name)

if __name__ == "__main__":
    # download real images
    for name, url in to_download.items():
        download_and_save(name, url)

    # create placeholders for the added items
    for name, label in placeholders.items():
        make_placeholder(name, label)

    print("All done. Check 'outfits' and 'thumbs' folders.")
