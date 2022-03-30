import csv
import json
from collections import OrderedDict

video_list = ['1','2','3','4','6']
roi_size_list = ['120','160','200']
for video_num in video_list:
    for roi_size in roi_size_list:
        frame_dict = OrderedDict()
        with open('./S20_Model_Log/test_VIRAT_'+video_num+'_'+roi_size+'.log', 'r') as log_file:
            rdr = csv.reader(log_file, delimiter=',')
            for line in rdr:
                if len(line) > 4:
                    box_list = []
                    i = 3
                    while i < len(line):
                        box_list.append(line[i:i+6])
                        i += 6
                    frame_dict[line[1]] = box_list

        with open('log_'+video_num+'_'+roi_size+'.json', 'w') as log_json:
            json.dump(frame_dict, log_json)
            