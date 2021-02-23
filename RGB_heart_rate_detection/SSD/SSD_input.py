# coding=utf-8
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from SSD.ssd_utils import BBoxUtility


priors = pickle.load(open('./SSD/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(2, priors)


def process_head_region(model, img):


    inputs = preprocess_input(np.array(img).astype(float))
    inputs = inputs[np.newaxis, :, :, :]
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    if results[0] == []:
        return [], [], [], [], [], []

    # Parse the outputs.
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    score = []
    label = []
    for i in range(top_conf.shape[0]):
        xmin.append((int(round(top_xmin[i] * img.shape[1]))))
        ymin.append(int(round(top_ymin[i] * img.shape[0])))
        xmax.append(int(round(top_xmax[i] * img.shape[1])))
        ymax.append(int(round(top_ymax[i] * img.shape[0])))
        score.append(top_conf[i])
        label.append(int(top_label_indices[i]))
        #         label_name = voc_classes[label - 1]
        # display_txt = 'T-Face, {:0.2f}, {}'.format(score, label)
        # coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        # color = colors[label]
        # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    return xmin, ymin, xmax, ymax, score, label