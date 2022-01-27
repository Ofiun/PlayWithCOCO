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
            file_name = filename_dict[image_id]
            if file_name not in file_seg_dict:
                file_seg_dict[file_name] = [[annotation['category_id'], annotation['bbox']]]
            else:
                file_seg_dict[file_name].append([annotation['category_id'], annotation['bbox']])

    with open('image-seginfo.json', 'w') as f:
        json.dump(file_seg_dict, f, indent="\t")

if __name__ == "__main__":
    main()
    