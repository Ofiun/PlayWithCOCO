import os
import json

def main():
    file_seg_dict = {}

    with open('./instances_val2017.json', 'r') as instances_val:
        js = json.load(instances_val)
        filename_dict = {}
        images = js['images']
        for image in images:
            filename_dict[image['id']] = image['file_name']

        annotations = js['annotations']
        for annotation in annotations:
            image_id = annotation['image_id']
            seg_id = annotation['id']
            file_name = filename_dict[image_id]
            file_name_no_ext = os.path.splitext(file_name)[0]
            seg_file_name = file_name_no_ext+'_'+str(seg_id)+'.jpg'
            bbox = annotation['bbox']
            file_seg_dict[seg_file_name] = bbox

    with open('roi_bbox.json', 'w') as f:
        json.dump(file_seg_dict, f, indent="\t")

if __name__ == "__main__":
    main()
    