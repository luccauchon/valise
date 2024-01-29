from flask import Flask, request, jsonify
from multiprocessing import freeze_support
import base64
import pathlib
from PIL import Image
from io import BytesIO
import time
import os
import numpy as np
from flask import Flask, session
from flask import Flask, send_file, request
from flask_session import Session
import cv2
import io
import random
import uuid
import os, fnmatch
from multiprocessing import Process, Queue, Value
import multiprocessing
from utils import scan_for_jpeg_and_png
from datetime import datetime
from functools import lru_cache
from PIL import ImageFont, ImageDraw, Image
from datetime import timedelta


# https://stackoverflow.com/questions/14888799/disable-console-messages-in-flask-server
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from loguru import logger

app = Flask(__name__)
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

GLOBAL_DATA = {}


def _generate_image(image_id=None):
    # Save the image to an in-memory file (BytesIO)
    image_io = io.BytesIO()
    w, h = GLOBAL_DATA['w'], GLOBAL_DATA['h']

    formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if image_id < 0 or image_id > len(GLOBAL_DATA['images_data']) - 1:
        image = Image.new('RGB', (w, h), color='red')
    else:
        file_path = GLOBAL_DATA['images_data'][image_id][0]
        # print(f'[{formatted_datetime}]REMOVE ME -- {file_path=}  {image_id=}', flush=True)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
        image = Image.fromarray(image)

    image.save(image_io, format='JPEG')
    return image_io.getvalue()


# def _inference_image(image_id=None, txt=None):  # FIXME REMOVE ME
#     # Save the image to an in-memory file (BytesIO)
#     image_io = io.BytesIO()
#     w, h = GLOBAL_DATA['w'], GLOBAL_DATA['h']
#     import sys
#     formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#     if image_id < 0 or image_id > len(GLOBAL_DATA['images_data']) - 1:
#         image = Image.new('RGB', (w, h), color='red')
#     else:
#         file_path = GLOBAL_DATA['images_data'][image_id][0]
#         # print(f'[{formatted_datetime}]REMOVE ME -- {file_path=}  {image_id=}', flush=True)
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
#         image = Image.fromarray(image)
#     sys.exit(0)
#     image.save(image_io, format='JPEG')
#     return image_io.getvalue()


# @app.route('/posteinformatique/listeimages', methods=['POST'])
# def a360_pi_liste_images(): # FIXME REMOVE ME
#     try:
#         GLOBAL_DATA['counter'] += 1
#         _base_url = GLOBAL_DATA['base_url'] + 'posteinformatique/get_image?'
#         batch_number = request.get_json()['batch_number']
#         batch_size = GLOBAL_DATA['batch_size']
#         image_key = list(GLOBAL_DATA['images_data'].keys())
#
#         urls_data, names_data = "[", "["
#         for j in range(batch_number * batch_size, batch_number * batch_size + batch_size):
#             the_key = image_key[j]
#             fullpath, filename = GLOBAL_DATA['images_data'][the_key]
#             urls_data += f"\"{_base_url}imgid={the_key}\""
#             names_data += f"\"{filename}\""
#             if j < (batch_number * batch_size + batch_size) - 1:
#                 urls_data += ","
#                 names_data += ","
#         urls_data += "]"
#         names_data += "]"
#
#         return jsonify({'batch_size': batch_size, 'number_batches': (len(image_key) // batch_size) - 1,
#                         'images_url': urls_data, 'images_name': names_data})
#     except Exception as e:
#         return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/posteinformatique/listeimagespreview2', methods=['POST'])
def a360_pi_liste_imagespreview2():
    try:
        GLOBAL_DATA['counter'] += 1
        _base_url = GLOBAL_DATA['base_url'] + 'posteinformatique/get_image?'
        image_key = list(GLOBAL_DATA['images_data'].keys())
        urls_data, names_data = "[", "["
        for j in range(0, len(image_key)):
            the_key = image_key[j]
            fullpath, filename = GLOBAL_DATA['images_data'][the_key]
            urls_data += f"\"{_base_url}imgid={the_key}\""
            names_data += f"\"{filename}\""
            if j < len(image_key) - 1:
                urls_data += ","
                names_data += ","
        urls_data += "]"
        names_data += "]"

        return jsonify({'batch_size': 1, 'number_batches': len(image_key) - 1,
                        'images_url': urls_data, 'images_name': names_data})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@lru_cache(maxsize=128)  # Setting maxsize to None means the cache can grow without bound
def _local_get_image(image_id):
    payload = _generate_image(image_id)
    return payload


@app.route('/posteinformatique/get_image_name', methods=['POST'])
def get_image_name():
    image_id = int(request.args.get('imgid', -1))
    file_path = GLOBAL_DATA['images_data'][image_id][0]
    return jsonify({'filename': pathlib.PurePath(file_path).name})


@app.route('/posteinformatique/get_image', methods=['GET', 'POST'])
def get_image():
    image_id = int(request.args.get('imgid', -1))
    payload = _local_get_image(image_id)
    if request.method == 'GET':
        return send_file(io.BytesIO(payload), mimetype='image/jpeg')
    elif request.method == 'POST':
        return send_file(io.BytesIO(payload), mimetype='image/jpeg')


def serialize_image(image_array):
    # Convert NumPy array to a format suitable for serialization
    _, buffer = cv2.imencode('.png', image_array)
    image_bytes = buffer.tobytes()

    # Encode the image bytes using base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    return encoded_image


@app.route('/posteinformatique/do_inference_for_image', methods=['POST'])
def do_inference_for_image():
    image_id = int(request.args.get('imgid', -1))
    magic_number = int(request.args.get('magicnumber', -1))
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}]  -- {magic_number=} -- {image_id=}', flush=True)

    job_id = f'{int(uuid.uuid4())}'
    in__shared = GLOBAL_DATA['in__shared']
    file_path = GLOBAL_DATA['images_data'][image_id][0]
    try:
        w, h = GLOBAL_DATA['w'], GLOBAL_DATA['h']
        in__shared.put_nowait({'image_id': image_id, 'job_id': job_id, 'file_path': file_path, 'h': h, 'w': w})
    except:
        pass

    mask = serialize_image(np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8))
    return jsonify({'job_id': job_id, 'result_ready': 0, 'mask': mask})


@app.route('/posteinformatique/get_inference', methods=['POST'])
def get_inference():
    job_id, result_ready, mask = request.get_json()['job_id'], 0, np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    out__shared = GLOBAL_DATA['out__shared']
    resultats_inference = GLOBAL_DATA['resultats_inference']

    # Va chercher des réponses
    try:
        while True:
            response = out__shared.get_nowait()
            resultats_inference.update({response['job_id']: response})
    except:
        pass

    # logger.debug(f'Got {len(resultats_inference)} elements waiting')

    # Chercher pour notre réponse
    if job_id in resultats_inference:
        result_ready, mask = 1, resultats_inference[job_id]['mask']
        del resultats_inference[job_id]

    # Cleanup
    max_timeout, to_be_deleted = timedelta(days=0, hours=0, minutes=15, seconds=0), []
    for k, v in resultats_inference.items():
        dt = datetime.now() - v['timestamp']
        if dt > max_timeout:
            to_be_deleted.append(k)
    for k in to_be_deleted:
        logger.warning(f'Deleting {k}')
        del resultats_inference[k]

    return jsonify({'job_id': job_id, 'result_ready': result_ready, 'mask': serialize_image(mask)})


@app.route('/posteinformatique/do_inference_with_image', methods=['POST'])
def do_inference_with_image():
    # Get the JSON payload
    data = request.get_json()

    # Retrieve the base64-encoded image from the payload
    base64_image = data.get('image', '')

    # Decode base64 string to binary
    binary_image = base64.b64decode(base64_image)

    # Create an in-memory file-like object
    image_io = BytesIO(binary_image)

    # Open the image using PIL
    # image = Image.open(image_io)
    # image.save("C:\\Users\\cj3272\\Pictures\\new_example.jpg")
    h, w = np.array(Image.open(image_io)).shape[:2]

    job_id = f'{int(uuid.uuid4())}'

    in__shared = GLOBAL_DATA['in__shared']
    try:
        in__shared.put_nowait({'image_bytes': image_io, 'job_id': job_id, 'h': h, 'w': w})
    except:
        pass

    mask = serialize_image(np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8))
    return jsonify({'job_id': job_id, 'result_ready': 0, 'mask': mask})


def controller_processor(in__shared, out__shared, ):
    while True:
        try:
            payload = in__shared.get(timeout=1)
        except Exception as e:
            continue
        logger.debug(f'[Controller]\nProcessing {payload}')

        if 'image_id' in payload:
            image_id, job_id, file_path, h, w = payload['image_id'], payload['job_id'], payload['file_path'], payload['h'], payload['w']
            assert os.path.exists(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
            image = Image.fromarray(image)
        else:
            assert 'image_bytes' in payload
            image_io, job_id, h, w = payload['image_bytes'], payload['job_id'], payload['h'], payload['w']
            image = Image.open(image_io)
            image = image.resize((w, h), Image.ANTIALIAS)

        # Do some work
        height, width = np.array(image).shape[:2]
        mask = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Define rectangle parameters (x, y, width, height) for each corner
        rectangles = [
            (0, 0, 100, 100),  # Top-left corner
            (mask.shape[1] - 100, 0, 100, 100),  # Top-right corner
            (0, mask.shape[0] - 100, 100, 100),  # Bottom-left corner
            (mask.shape[1] - 100, mask.shape[0] - 100, 100, 100)  # Bottom-right corner
        ]
        # Draw rectangles on the image
        for rect in rectangles:
            x, y, width, height = rect
            mask[y:y + height, x:x + width, :] = [255, 0, 0]  # White color

        # Retourne la réponse
        while True:
            try:
                out__shared.put_nowait({'job_id': job_id, 'mask': mask, 'timestamp':datetime.now()})
                break
            except:
                pass


if __name__ == '__main__':
    freeze_support()

    # TODO à refactorer
    root_directory = 'D:\\dataset\\val2017'

    jpeg_files = scan_for_jpeg_and_png(root_directory)
    images_data = {}
    for e, a_file in enumerate(jpeg_files):
        images_data.update({e: (a_file, pathlib.PurePath(a_file).name)})
    if jpeg_files:
        print(f"{len(images_data)} JPEG files found")

    GLOBAL_DATA.update({'images_data': images_data})
    GLOBAL_DATA.update({'counter': 0,
                        # 'batch_size': 15,
                        'w': 1024, 'h': 768})  # Pour éviter de transférer trop de données
    GLOBAL_DATA.update({'base_url': 'http://127.0.0.1:5000/'})  # L'adresse du serveur; aussi dans le client
    GLOBAL_DATA.update({'resultats_inference': {}})  # Contient les résultats des inférences

    in__shared, out__shared = Queue(32), Queue(32)
    Process(target=controller_processor, args=(in__shared, out__shared,)).start()
    GLOBAL_DATA['in__shared'] = in__shared
    GLOBAL_DATA['out__shared'] = out__shared

    # # https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
