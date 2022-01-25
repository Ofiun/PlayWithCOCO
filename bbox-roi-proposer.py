#Cropping image
#https://stackoverflow.com/questions/6456479/python-pil-how-to-save-cropped-image
#https://stackoverflow.com/questions/9983263/how-to-crop-an-image-using-pil
#https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=crop#PIL.Image.Image.crop
#Removing Extenstion
#https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python
#COCO bbox
#https://github.com/cocodataset/cocoapi/issues/102

import os
import json
from PIL import Image
from os.path import isfile, join

src_dir = '../val2017/'
dst_dir = '../val2017_roi/'

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

with open('./instances_val2017.json', 'r') as coco:
    js = json.load(coco)
    categories = js['categories']
    for category in categories:
        class_name = category['name']
        createFolder(dst_dir+class_name)

image_dict = {}

def crop_and_save(file_name, class_name):
    img = Image.open(src_dir+file_name)
    bbox_tuples = image_dict[file_name][class_name]
    for bbox_tuple in bbox_tuples:
        bbox_t = tuple(bbox_tuple[0])
        bbox = (bbox_t[0], bbox_t[1], bbox_t[0]+bbox_t[2], bbox_t[1]+bbox_t[3])
        # The right can also be represented as (left+width)
        # and lower can be represented as (upper+height).
        # im_crop = im.crop((left, upper, right, lower))
        cropped_img = img.crop(bbox)
        file_name_no_ext = os.path.splitext(file_name)[0]
        dst_file_path = dst_dir+class_name+'/'+file_name_no_ext+'_'+str(bbox_tuple[1])+'.jpg'
        if not os.path.exists(dst_file_path):
            cropped_img.save(dst_file_path)

with open('./instances_val2017.json', 'r') as instances_val:
    js = json.load(instances_val)
    filename_dict = {}
    images = js['images']
    for image in images:
        filename_dict[image['id']] = image['file_name']

    category_dict = {}
    categories = js['categories']
    for category in categories:
        class_id = category['id']
        class_name = category['name']
        category_dict[class_id] = class_name
    
    annotations = js['annotations']
    for annotation in annotations:
        image_id = annotation['image_id']
        file_name = filename_dict[image_id]
        category_id = annotation['category_id']
        class_name = category_dict[category_id]
        seg_id = annotation['id']
        bbox = annotation['bbox']
        if file_name not in image_dict:
            inner_dict = {}
            inner_dict[class_name] = [(bbox, seg_id)]
            image_dict[file_name] = inner_dict
        elif class_name not in image_dict[file_name]:
            image_dict[file_name][class_name] = [(bbox, seg_id)]
        else:
            image_dict[file_name][class_name].append((bbox, seg_id))

    file_names = [f for f in os.listdir(src_dir) if isfile(join(src_dir, f))]
    for file_name in file_names:
        if file_name in image_dict:
            class_list = image_dict[file_name]
            for class_name in class_list:
                crop_and_save(file_name, class_name)
        else:
            print(file_name + " has no detected objects.")