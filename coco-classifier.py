#Listing files in a directory
#https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
#Create folder
#https://data-make.tistory.com/170
#Copy files
#https://pynative.com/python-copy-files-and-directories/

import os
from os.path import isfile, join
import json
import shutil

src_dir = '../val2017/'
dst_dir = '../val2017_classes/'

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

with open('./instances_val2017.json', 'r') as coco:
    js = json.load(coco)
    categories = js['categories']
    for category in categories:
        class_name = category['name']
        createFolder(dst_dir+class_name)

def copy(src_path, dst_path):
    shutil.copy(src_path, dst_path)

with open('./categories.json', 'r') as cat:
    cat_dict = json.load(cat)

    file_names = [f for f in os.listdir(src_dir) if isfile(join(src_dir, f))]
    for file_name in file_names:
        if file_name in cat_dict:
            class_list = cat_dict[file_name]
            for class_name in class_list:
                dst_path = dst_dir+class_name+'/'+file_name
                if not os.path.exists(dst_path):
                    copy(src_dir+file_name, dst_path)
        else:
            print(file_name + " has no detected objects.")
    