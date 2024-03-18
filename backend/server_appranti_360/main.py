from flask import Flask, request, jsonify
from multiprocessing import freeze_support
from tqdm import tqdm
import base64
from einops import rearrange, reduce, repeat
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
import sys
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


def _build_preview():
    _base_url = GLOBAL_DATA['base_url'] + 'posteinformatique/get_image?'
    image_key = list(GLOBAL_DATA['images_data'].keys())
    urls_data, names_data = "[", "["
    for j in tqdm(range(0, len(image_key))):
        the_key = image_key[j]
        fullpath, filename = GLOBAL_DATA['images_data'][the_key]
        urls_data += f"\"{_base_url}imgid={the_key}\""
        names_data += f"\"{filename}\""
        if j < len(image_key) - 1:
            urls_data += ","
            names_data += ","
    urls_data += "]"
    names_data += "]"

    return {'batch_size': 1, 'number_batches': len(image_key) - 1, 'images_url': urls_data, 'images_name': names_data}


@app.route('/posteinformatique/listeimagespreview2', methods=['POST'])
def a360_pi_liste_imagespreview2():
    try:
        GLOBAL_DATA['counter'] += 1

        result = GLOBAL_DATA['preview']

        return jsonify(result)
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
    '''
    Retourne l'image demandée
    '''
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
    '''
    Demande une inférence sur une image disponible sur le disque
    :return:
    '''
    image_id = int(request.args.get('imgid', -1))
    magic_number = int(request.args.get('magicnumber', -1))
    model_id = request.args.get('modeleid', 'mock')
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}]  -- {magic_number=} -- {image_id=} -- {model_id=}', flush=True)

    job_id = f'{int(uuid.uuid4())}'
    in__shared = get_in__shared(model_id)
    file_path = GLOBAL_DATA['images_data'][image_id][0]
    try:
        in__shared.put_nowait({'image_id': image_id, 'job_id': job_id, 'file_path': file_path})
    except:
        pass

    mask = serialize_image(np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8))
    return jsonify({'job_id': job_id, 'result_ready': 0, 'mask': mask})


def get_in__shared(model_id):
    for k, v in GLOBAL_DATA.items():
        if k.endswith(f'__{model_id}'):
            in__shared = v['in__shared']
            return in__shared
    return None


def get_out__shared(model_id):
    for k, v in GLOBAL_DATA.items():
        if k.endswith(f'__{model_id}'):
            out__shared = v['out__shared']
            return out__shared
    return None


@app.route('/posteinformatique/get_inference', methods=['POST'])
def get_inference():
    job_id, model_id, result_ready, mask = request.get_json()['job_id'], request.get_json()['model_id'], 0, np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    out__shared, resultats_inference = get_out__shared(model_id), GLOBAL_DATA['resultats_inference']

    # Va chercher des réponses reliée au modèle spécifié et stock la réponse
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


def letter_to_matrix(letter, size=(20, 20), font_size=96):
    # Create a blank image
    image = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(image)

    # Load the font with the specified size
    font = ImageFont.truetype("arial.ttf", font_size)

    # Calculate the position to center the letter
    num_lines = 1
    text_width = draw.textlength(letter, font=font)
    text_height = font.size * num_lines  # Adjust num_lines for multiline text
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    # Draw the letter on the image
    draw.text((x, y), letter, font=font, fill='white')

    # Convert the image to a numpy array
    matrix = np.array(image)

    return matrix


def worker_processor_mock(configuration, ):
    logger.debug(f'[{os.getpid()}] Starting model {configuration["description"]}')
    in__shared, out__shared = configuration['in__shared'], configuration['out__shared']
    logger.debug(f'[{os.getpid()}] Serving model {configuration["description"]}')
    while True:
        try:
            payload = in__shared.get(timeout=1)
        except Exception as e:
            continue
        logger.debug(f'[{os.getpid()}] Processing {payload}')

        if 'image_id' in payload:  # This means we read the data from disk
            image_id, job_id, file_path, h_fe, w_fe, = payload['image_id'], payload['job_id'], payload['file_path'], configuration['h_frontend'], configuration['w_frontend']
            assert os.path.exists(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(w_fe, h_fe), interpolation=cv2.INTER_LANCZOS4)
            image = Image.fromarray(image)
        else:  # We received the data from the client
            assert 'image_bytes' in payload
            image_io, job_id, h_fe, w_fe = payload['image_bytes'], payload['job_id'], configuration['h_frontend'], configuration['w_frontend']
            image = Image.open(image_io)
            image = image.resize((w_fe, h_fe), Image.ANTIALIAS)

        # Do some work with model
        height_fe, width_fe = configuration['h_frontend'], configuration['w_frontend']
        mask = np.random.randint(0, 256, (height_fe, width_fe, 3), dtype=np.uint8)

        # Define rectangle parameters (x, y, width, height) for each corner
        rectangles = [
            (0, 0, 100, 100),  # Top-left corner
            (mask.shape[1] - 100, 0, 100, 100),  # Top-right corner
            (0, mask.shape[0] - 100, 100, 100),  # Bottom-left corner
            (mask.shape[1] - 100, mask.shape[0] - 100, 100, 100)  # Bottom-right corner
        ]
        # Draw rectangles on the image
        for rect in rectangles:
            x2, y2, width2, height2 = rect
            mask[y2:y2 + height2, x2:x2 + width2, :] = [255, 0, 0]  # White color

        # Print model's name in image
        tmp = letter_to_matrix(letter="Hello world! (from server)", size=(width_fe, height_fe))
        mask = cv2.addWeighted(mask, 1., tmp, 1, 0)

        # Retourne la réponse
        while True:
            try:
                out__shared.put_nowait({'job_id': job_id, 'mask': mask, 'timestamp':datetime.now()})
                break
            except:
                pass


def worker_processor_SAM(configuration):
    logger.debug(f'[{os.getpid()}] Starting model {configuration["description"]}')
    in__shared, out__shared = configuration['in__shared'], configuration['out__shared']

    for path_to_add in configuration['path_code_source']:
        sys.path.append(path_to_add)
    from segment_anything import sam_model_registry
    from importlib import import_module
    import torch
    import gc
    config = configuration['model_parameters']
    couleurs = configuration['couleurs']
    cat_id__2__cat_name = config['cat_id__2__cat_name']
    ckpt_path = os.path.join(config['weights'], 'ckpt.pt')
    logger.debug(f"[{os.getpid()}] Loading model {ckpt_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=config["device"])
    w_img__in, h_img__in = config['w_img__in'], config['h_img__in']
    assert w_img__in == h_img__in and config['w_img__out'], config['h_img__out']
    image_size = w_img__in
    factor_out = (int(config['h_img__out'] / h_img__in), int(config['w_img__out'] / w_img__in))
    num_classes = config['num_classes']
    # create the model
    sam, img_embedding_size = sam_model_registry['vit_h'](image_size=image_size,
                                                          num_classes=num_classes - 1,  # SAM en ajoute une
                                                          checkpoint=config['sam_checkpoint'],
                                                          pixel_mean=[123.675, 116.28, 103.53],
                                                          pixel_std=[58.395, 57.12, 57.375])
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam_model=sam, r=config['rank'], lora_layer=None, ce_weight=None, use_cl_dice=False, image_size=image_size, factor_out=factor_out)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model.eval()
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    logger.debug(f"[{os.getpid()}]  --> {iter_num=}   {best_val_loss=}")
    logger.debug(f'[{os.getpid()}] Serving model {configuration["description"]}')
    while True:
        try:
            payload = in__shared.get(timeout=1)
        except Exception as e:
            continue
        logger.debug(f'[{os.getpid()}] Processing {payload}')

        if 'image_id' in payload:  # This means we read the data from disk
            image_id, job_id, file_path, h_in, w_in, = payload['image_id'], payload['job_id'], payload['file_path'], config['h_img__in'], config['w_img__in']
            assert os.path.exists(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(w_in, h_in), interpolation=cv2.INTER_LANCZOS4)
        else:  # We received the data from the client
            assert 'image_bytes' in payload
            image_io, job_id, h_in, w_in = payload['image_bytes'], payload['job_id'], configuration['h_img__in'], configuration['w_img__in']
            image = Image.open(image_io)
            image = image.resize((w_in, h_in), Image.ANTIALIAS)

        # Do some work with model
        x, images_x = image, []
        x = rearrange(x, 'h w c -> c h w')
        images_x.append(x)
        x = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in images_x])
        if config["device"] == 'cuda':
            x = x.cuda()
        t1 = time.time()
        y_pred = model(batched_input=x)
        t2 = time.time()
        logger.debug(f'[{os.getpid()}] Inference done in {t2-t1:0.4} sec')

        # Dessine les prédictions
        height_fe, width_fe = configuration['h_frontend'], configuration['w_frontend']
        the_y_pred = torch.zeros(config['h_img__out'], config['w_img__out'], 3, dtype=torch.float32, device=config["device"])
        for class_id in range(1, num_classes):
            tmp_pred_y = torch.stack((y_pred[0][class_id] * 1,) * 3, dim=2)
            assert np.prod(tmp_pred_y.shape) == torch.count_nonzero(tmp_pred_y == 0) + torch.count_nonzero(tmp_pred_y == 1)
            the_color = couleurs[cat_id__2__cat_name[class_id]][:-1]
            tmp_pred_y = tmp_pred_y * torch.tensor(the_color, dtype=torch.float32, device=config["device"])
            the_y_pred = the_y_pred + tmp_pred_y
        the_y_pred = the_y_pred.cpu().numpy()
        mask = cv2.resize(the_y_pred, dsize=(width_fe, height_fe), interpolation=cv2.INTER_LANCZOS4)

        # Retourne la réponse
        while True:
            try:
                out__shared.put_nowait({'job_id': job_id, 'mask': mask, 'timestamp': datetime.now()})
                break
            except:
                pass


if __name__ == '__main__':
    freeze_support()

    # TODO à refactorer
    # Les images
    images_dir = 'D:\\dataset\\val2017'
    # Le code source des modèles
    path_code_source = [r'D:\PyCharmProjects\server_appranti_360\src_from_ireq\SAMed_h']
    sam_checkpoint = r'D:\PyCharmProjects\server_appranti_360\src_from_ireq\SAMed_h\appranti-360\checkpoints\sam_vit_h_4b8939.pth'
    weights_dir = r"D:\PyCharmProjects\server_appranti_360\2024.03.14.Demo\\"

    jpeg_files = scan_for_jpeg_and_png(images_dir)
    images_data = {}
    for e, a_file in enumerate(jpeg_files):
        images_data.update({e: (a_file, pathlib.PurePath(a_file).name)})
    if jpeg_files:
        print(f"{len(images_data)} JPEG files found")

    GLOBAL_DATA.update({'images_data': images_data})
    GLOBAL_DATA.update({'counter': 0,
                        'w': 4096, 'h': 3072})  # (h,w) des images lues sur le disque. C'est pour éviter de transférer trop de données ;)
    GLOBAL_DATA.update({'base_url': 'http://127.0.0.1:5000/'})  # L'adresse du serveur; aussi dans le client
    GLOBAL_DATA.update({'preview': _build_preview()})  # Construit la liste des images disponibles pour le client
    GLOBAL_DATA.update({'resultats_inference': {}})  # Contient les résultats des inférences

    couleurs = {
        'C': (255, 0, 0, 'Red'),'CP': (0, 255, 0, 'Green'),'D': (0, 0, 255, 'Blue'),'DE': (255, 255, 0, 'Yellow'),
        'DEP': (255, 0, 255, 'Magenta'),'E': (0, 255, 255, 'Cyan'),'EC': (128, 0, 0, 'Maroon'),'F': (0, 128, 0, 'Olive'),
        'FD': (0, 0, 128, 'Navy'),'FI': (128, 128, 0, 'Olive Green'),'FJ': (128, 0, 128, 'Purple'),'FJM': (0, 128, 128, 'Teal'),  # sarcelle
        'FP': (255, 165, 0, 'Orange'),'NC': (128, 128, 128, 'Gray'),'TC': (255, 255, 255, 'White'),}
    # Chaque modèle à ses propres canaux et process
    ###########################################################################
    # 10CL
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__10CL': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle 10 classes (2024.03.15)',
                                             'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                             'couleurs': couleurs,
                                             'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 10+1,
                                                                  'cat_id__2__cat_name': {1: 'D',2: 'C',3: 'TC',4: 'FP',5: 'E',6: 'DE',7: 'EC',8: 'NC',9: 'CP',10: 'DEP'},
                                                                  'sam_checkpoint': sam_checkpoint,
                                                                  'weights': os.path.join(weights_dir, '10CL')}}})

    ###########################################################################
    # 5CL
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__5CL': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle 5 classes (2024.03.15)',
                                             'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                             'couleurs': couleurs,
                                             'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 5 + 1,
                                                                  'cat_id__2__cat_name': {1: 'F', 2: 'FD', 3: 'FI', 4: 'FJ', 5: 'FJM'},
                                                                  'sam_checkpoint': sam_checkpoint,
                                                                  'weights': os.path.join(weights_dir, '5CL')}}})

    ###########################################################################
    # Anomalie
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__anomalie': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle Anomalie (2024.03.15)',
                                            'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                            'couleurs': couleurs,
                                            'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 1 + 1,
                                                                 'cat_id__2__cat_name': {1: 'TC'},  # Juste pour avoir la couleur
                                                                 'sam_checkpoint': sam_checkpoint,
                                                                 'weights': os.path.join(weights_dir, 'anomalie')}}})

    ###########################################################################
    # D
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__D': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle D (2024.03.15)',
                                                 'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                                 'couleurs': couleurs,
                                                 'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 1 + 1,
                                                                      'cat_id__2__cat_name': {1: 'D'},
                                                                      'sam_checkpoint': sam_checkpoint,
                                                                      'weights': os.path.join(weights_dir, 'D')}}})

    ###########################################################################
    # F
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__F': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle F (2024.03.15)',
                                          'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                          'couleurs': couleurs,
                                          'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 1 + 1,
                                                               'cat_id__2__cat_name': {1: 'F'},
                                                               'sam_checkpoint': sam_checkpoint,
                                                               'weights': os.path.join(weights_dir, 'F')}}})

    ###########################################################################
    # C
    ###########################################################################
    GLOBAL_DATA.update({'worker_sam__C': {'in__shared': Queue(32), 'out__shared': Queue(32), 'path_code_source': path_code_source, 'description': 'Modèle C (2024.03.15)',
                                          'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h'],
                                          'couleurs': couleurs,
                                          'model_parameters': {'w_img__in': 500, 'h_img__in': 500, 'w_img__out': 1500, 'h_img__out': 1500, 'device': 'cpu', 'rank': 4, 'num_classes': 1 + 1,
                                                               'cat_id__2__cat_name': {1: 'C'},
                                                               'sam_checkpoint': sam_checkpoint,
                                                               'weights': os.path.join(weights_dir, 'C')}}})

    ###########################################################################
    # mock
    ###########################################################################
    GLOBAL_DATA.update({'worker_dev__mock': {'in__shared': Queue(32), 'out__shared': Queue(32), 'description': '123', 'w_frontend': GLOBAL_DATA['w'], 'h_frontend': GLOBAL_DATA['h']}})

    for k, v in GLOBAL_DATA.items():
        if k.startswith('worker_sam__'):
            configuration = v
            Process(target=worker_processor_SAM, args=(configuration, )).start()
        if k.startswith('worker_dev__'):
            configuration = v
            Process(target=worker_processor_mock, args=(configuration, )).start()

    # # https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
