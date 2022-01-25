#Getting greater Indicies
#https://www.geeksforgeeks.org/python-indices-of-numbers-greater-than-k/
#Zero-pad OpenCV
#https://linuxtut.com/en/540d3be3e570cbca644e/

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
class_num = 0
src_dir = '../val2017_roi/'

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_intersection_area(box1, box2):
    left = box1[0] if box1[0] > box2[0] else box2[0]
    right = box1[2] if box1[2] < box2[2] else box2[2]
    up = box1[1] if box1[1] > box2[1] else box2[1]
    down = box1[3] if box1[3] < box2[3] else box2[3]
    return (right - left) * (down - up)

def get_area(box):
    return (box[2]-box[0]) * (box[3]-box[1])

def get_iou(box1, box2):
    box1_area = get_area(box1)
    box2_area = get_area(box2)
    common_area = get_intersection_area(box1, box2)
    return common_area / (box1_area + box2_area - common_area)

def execute(target_num, target_pixel, img_dir, file_names, saved_model_loaded):
    i = 0
    no_bbox = 0
    no_person = 0
    iouSum = 0
    endIdx = len(file_names)-1
    while True:
        rand_idx = random.randint(0, endIdx)
        roi_name = file_names[rand_idx]
        original_image = cv2.imread(img_dir+roi_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        dims = original_image.shape
        larger_axis_idx = 0 if dims[0] > dims[1] else 1
        if dims[larger_axis_idx] < target_pixel:
            pass
        else: 
            smaller_axis_idx = 1 - larger_axis_idx
            larger_length = dims[larger_axis_idx]
            smaller_length = dims[smaller_axis_idx]
            dim_list = [0, 0]
            dim_list[larger_axis_idx] = target_pixel
            dim_list[smaller_axis_idx] = int(smaller_length * target_pixel / float(larger_length))
            img_raw = cv2.resize(original_image, (dim_list[1], dim_list[0]))
            height_pad = int((416-dim_list[0])/2)
            width_pad = int((416-dim_list[1])/2)
            img_pad = cv2.copyMakeBorder(img_raw, height_pad, 416-dim_list[0]-height_pad, width_pad, 416-dim_list[1]-width_pad, cv2.BORDER_CONSTANT, (255, 255, 255))
            img_data = img_pad / 255.

            cv2_gtb = [height_pad, width_pad, height_pad+dim_list[0], width_pad+dim_list[1]]
            nm_gtb = np.array(cv2_gtb) / 416

            images_data = []
            images_data.append(img_data)
            images_data = np.asarray(images_data).astype(np.float32)

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
            boxes_f = boxes.numpy()[0]
            scores_f = scores.numpy()[0]
            classes_f = classes.numpy()[0]
            iou_done = False
            bbox_found = False
            largest_box = None
            largest_area = 0
            for idx in range(0, len(scores_f)):
                if scores_f[idx] > 0.25 and classes_f[idx] == class_num:
                    box = boxes_f[idx]
                    area = get_area(box)
                    if largest_area < area:
                        largest_area = area
                        largest_box = box
                    iou_done = True
            if iou_done:
                iouSum += get_iou(largest_box, nm_gtb)
            if not iou_done:
                for idx in range(0, len(scores_f)):
                    if scores_f[idx] > 0.25:
                        no_person += 1
                        bbox_found = True
                        break
                if not bbox_found:
                    no_bbox +=1

            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(img_pad, pred_bbox)
            image = Image.fromarray(image.astype(np.uint8))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            createFolder('./'+class_name+'_'+str(target_pixel))
            cv2.imwrite('./'+class_name+'_'+str(target_pixel)+'/'+roi_name, image)
            i += 1
            if i == target_num:
                break
    return no_bbox, no_person, iouSum / target_num

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    input_size = FLAGS.size
    img_dir = src_dir+class_name+'/'
    
    file_names = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f))]

    target_pixels = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,150,200,250,300,416]
    target_num = 200
    for target_pixel in target_pixels:
        no_bbox_count, no_person_count, mIoU = execute(target_num, target_pixel, img_dir, file_names, saved_model_loaded)
        print("target pixel (",target_pixel,"): no bbox (",no_bbox_count,") | no person (",no_person_count,") | mIoU (",mIoU,")")
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
