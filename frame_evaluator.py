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

#video_list = ['1','2','3','4','6']
video_list = ['1']
roi_size_list = ['120','160','200','full']
window_sizes = [10, 20, 40, 80]
#window_sizes = [1]

def get_miou_list():
    for window_size in window_sizes:
        #video_dict = {}
        for video_num in video_list:
            #gt_dict = {}
            N = window_size
            with open('./data/gt_'+video_num+'_person.json','r') as gt_json:
                gt_dict = json.load(gt_json)
            #miou_dict = {}
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
                #miou_dict[str(roi_size)] = miou_list.tolist()
                def get_normal_result(miou_list, roi_size):
                    plt.plot(range(len(miou_list)), miou_list, label=roi_size)
                def get_zoomed_result(startFrame, frameWidth, miou_list, roi_size):
                    miou_list = miou_list[startFrame:startFrame+frameWidth]
                    plt.plot(range(startFrame, startFrame+frameWidth), miou_list, label=roi_size)
                get_normal_result(miou_list, roi_size)
                #get_zoomed_result(12200, 200, miou_list, roi_size)
            #video_dict[str(video_num)] = miou_dict
            plt.title('mIoU variations of Video '+video_num)
            plt.xlabel('Frame Number')
            plt.ylabel('mIoU')
            plt.legend()
            plt.savefig('./data/video_'+video_num+'_window_'+str(N)+'.png')
            plt.cla()
        '''with open('./miou_list.json', 'w') as w:
            json.dump(video_dict,w)'''

def get_variance_statistics():
    with open('./data/miou_list.json', 'r') as r:
        video_miou_dict = json.load(r)
    for video_num in video_list:
        vars = []
        print('video num', video_num)
        video_pxl_list = video_miou_dict[video_num]
        for i in range(len(roi_size_list)):
            pxl_size = roi_size_list[i]
            miou_list = video_pxl_list[pxl_size]
            mv = np.var(miou_list)
            print('pxl_size',pxl_size,'var :',mv)
            miou_list_non_zero = [e for e in miou_list if e != 0]
            mnv = np.var(miou_list_non_zero)
            print('pxl_size',pxl_size,'var :',mnv,' (non zero)')
            #vars.append((mv+mnv)/2)
            vars.append(mv)
        print('px120 / px200 :',vars[0] / vars[2])
        print('px160 / px200 :',vars[1] / vars[2])

get_miou_list()
#get_variance_statistics()
