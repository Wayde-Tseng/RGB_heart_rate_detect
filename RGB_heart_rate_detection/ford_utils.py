import cv2
import pickle
import re
import numpy as np
import json
# from termcolor import cprint
from PIL import Image
from random import randint
from time import strftime, localtime


def get_video_configs(video_reader):
    video_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(video_reader.get(cv2.CAP_PROP_FPS)))
    video_frame_num = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_num == -1:
        video_frame_num = 'webcam'

    configs = {'video_w': video_w, 'video_h': video_h, 'fps': fps, 'video_frame_num': video_frame_num}

    return configs


def video_writer(out_video_path, video_width, video_height, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30):
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (video_width, video_height))
    return writer


def video_writer_just_like_reader(out_video_path, video_reader):
    video_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(video_reader.get(cv2.CAP_PROP_FPS)))

    writer = video_writer(out_video_path, video_w, video_h, fps=fps)
    return writer


def opencv_to_PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    return image


def PIL_to_opencv(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def image_visualize(map):
    map = map - np.min(map)
    map = map / np.max(map)
    map = map * 255
    map = map.astype(np.uint8)

    return map


def printf(string, color='yellow', on_color=None, attrs=['bold'], **kwargs):
    """
    color:
        'grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
    on_color:
        'on_grey', 'on_red', 'on_green', 'on_yellow', 'on_blue', 'on_magenta', 'on_cyan', 'on_white'
    attrs:
        'bold', 'dark', 'underline', 'blink', 'reverse', 'concealed'
    """

    available_colors = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'random']
    available_on_colors = ['on_grey', 'on_red', 'on_green', 'on_yellow', 'on_blue', 'on_magenta', 'on_cyan', 'on_white',
                           None]

    assert (color in available_colors), 'color must be in {}'.format(available_colors)
    assert (on_color in available_on_colors), 'on_color must be in {}'.format(available_on_colors)

    if color == 'random':
        r = randint(0, len(available_colors) - 1)
        color = available_colors[r]

    cprint(string, color=color, on_color=on_color, attrs=attrs, **kwargs)


def save_pickle(pkl_path, data):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_npy(npy_path, data):
    np.save(npy_path, data)


def load_npy(npy_path):
    try:
        data = np.load(npy_path).item()
    except:
        data = np.load(npy_path, allow_pickle='true')
    return data


def save_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f)


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.loads(f.read())
    return json_data


def plot_background(image, x_min, y_min, x_max, y_max, color=(0, 0, 0)):
    image[y_min:y_max, x_min:x_max, :] = color
    return image


def update_image(image, window_name='image', delay_time=1, scale=1, limit_max_size=0):
    if scale != 1:
        h, w, _ = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    if limit_max_size:
        h, w, _ = image.shape
        m = max(h, w)
        if m > limit_max_size:
            scale = limit_max_size / m
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

    cv2.imshow(window_name, image)
    cv2.waitKey(delay_time)


def put_text(image, text, position, font_type=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(0, 255, 255),
             text_thickness=1, line_type=cv2.LINE_AA):
    # position must be tuple (x, y)
    image = cv2.putText(image, text, position, font_type, font_scale, text_color, text_thickness, line_type)
    return image


def custom_sort(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


def get_local_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())
