#Getting greater Indicies
#https://www.geeksforgeeks.org/python-indices-of-numbers-greater-than-k/
#Zero-pad OpenCV
#https://linuxtut.com/en/540d3be3e570cbca644e/

import enum
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
from os.path import isfile, join
import random

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

class_name = 'person'
src_dir = '../val2017_roi/'

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    img_dir = src_dir+class_name+'/'
    
    file_names = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f))]

    endIdx = len(file_names)-1
    target_num = 20
    target_pixel = 90
    i = 0
    flag = 0
    while True:
        rand_idx = random.randint(0, endIdx)
        original_image = cv2.imread(img_dir+file_names[rand_idx])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        dims = original_image.shape
        larger_axis_idx = 0 if dims[0] > dims[1] else 1
        if dims[larger_axis_idx] < target_pixel:
            pass
        else: 
            print(file_names[rand_idx])
            smaller_axis_idx = 1 - larger_axis_idx
            larger_length = dims[larger_axis_idx]
            smaller_length = dims[smaller_axis_idx]
            dim_list = [0, 0]
            dim_list[larger_axis_idx] = target_pixel
            dim_list[smaller_axis_idx] = int(smaller_length * target_pixel / larger_length)
            img_raw = cv2.resize(original_image, (dim_list[1], dim_list[0]))
            height_pad = int((416-dim_list[0])/2)
            width_pad = int((416-dim_list[1])/2)
            img_pad = cv2.copyMakeBorder(img_raw, height_pad, height_pad, width_pad, width_pad, cv2.BORDER_CONSTANT, (255, 255, 255))
            img_data = cv2.resize(img_pad, (416, 416))
            img_data = img_data / 255.

            images_data = []
            images_data.append(img_data)
            images_data = np.asarray(images_data).astype(np.float32)

            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            scores_f = scores.numpy()[0]
            classes_f = classes.numpy()[0]
            for idx in range(0, len(scores_f)):
                if scores_f[idx] > 0.25 and classes_f[idx] == 0:
                    flag += 1
                    break
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(img_pad, pred_bbox)
            # image = utils.draw_bbox(image_data*255, pred_bbox)
            image = Image.fromarray(image.astype(np.uint8))
            #image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            createFolder('./'+class_name+'_'+str(target_pixel))
            cv2.imwrite('./'+class_name+'_'+str(target_pixel)+'/'+file_names[rand_idx], image)
            i += 1
            if i == target_num:
                break
    print(flag)
    

    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
