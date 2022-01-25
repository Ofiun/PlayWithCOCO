import json

def main():
    image_dict = {}

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
            if file_name not in image_dict:
                image_dict[file_name] = [class_name]
            else:
                image_dict[file_name].append(class_name)

    with open('categories.json', 'w') as f:
        json.dump(image_dict, f, indent="\t")

if __name__ == "__main__":
    main()
    