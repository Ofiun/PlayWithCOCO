import json
import numpy as np
import matplotlib.pyplot as plt

def convert_log_box(log_box):
    return [int(p) for p in log_box[0:4]]

def convert_gt_box(gt_box):
    left, top, width, height = gt_box[1]
    return [left, top, left+width, top+height]
    
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
    if box1_area + box2_area - common_area == 0:
        return 0
    iou = common_area / (box1_area + box2_area - common_area)
    if iou > 1:
        #print('iou error occurred')
        iou = 0
    return iou

def get_miou(log_boxes, gt_boxes):
    log_boxes = [convert_log_box(log_box) for log_box in log_boxes]
    gt_boxes = [convert_gt_box(gt_box) for gt_box in gt_boxes]
    ious = []
    for gt_box in gt_boxes:
        best_log_box = None
        best_iou = 0
        for log_box in log_boxes:
            iou = get_iou(log_box, gt_box)
            if iou > best_iou:
                best_log_box = log_box
                best_iou = iou
        ious.append(best_iou)
        if best_log_box is not None:
            log_boxes.remove(best_log_box)
    for _ in range(len(log_boxes)):
        ious.append(0)
    return sum(ious)/len(ious)

video_list = ['1','2','3','4','6']
roi_size_list = ['120','160','200']

for video_num in video_list:
    gt_dict = {}
    N=1
    with open('./data/gt_'+video_num+'_person.json','r') as gt_json:
        gt_dict = json.load(gt_json)
    for roi_size in roi_size_list:
        log_dict = {}
        with open('./data/log_'+video_num+'_'+roi_size+'.json','r') as log_json:
            log_dict = json.load(log_json)

        gt_keys = list(gt_dict.keys())
        log_keys = list(log_dict.keys())
        gt_end = gt_keys[-1]
        miou_list = []
        frames = list(range(int(gt_end)+1))
        for i in frames:
            frame_key = str(i)
            if frame_key in log_keys and frame_key in gt_keys:
                miou = get_miou(log_dict[frame_key], gt_dict[frame_key])
                miou_list.append(miou)
            elif frame_key in log_keys and frame_key not in gt_keys:
                # FP
                miou_list.append(0)
            elif frame_key not in log_keys and frame_key in gt_keys:
                # FN
                miou_list.append(0)
            else:
                miou_list.append(0)
        miou_list = np.convolve(miou_list, np.ones(N)/N, mode='valid')
        plt.plot(range(len(miou_list)), miou_list, label=roi_size)
    plt.title('mIoU variations of Video '+video_num)
    plt.xlabel('Frame Number')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig('./data/video_'+video_num+'_window_'+str(N)+'.png')
    plt.cla()