import os
from PIL import Image
from tqdm import tqdm


def is_jpeg_or_is_png(file_path):
    try:
        with Image.open(file_path) as img:
            return img.format == 'JPEG' or img.format == 'PNG'
    except (IOError, OSError):
        return False


def scan_for_jpeg_and_png(directory):
    jpeg_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_jpeg_or_is_png(file_path):
                jpeg_files.append(file_path)

    return jpeg_files
