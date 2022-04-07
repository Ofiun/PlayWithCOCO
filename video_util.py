import cv2
import json

pxl_sizes = ['120', '160', '200']
video_num = '4'

for pxl_size in pxl_sizes:
    cap = cv2.VideoCapture('./VIRAT_S_00000'+video_num+'.mp4')
    with open('log_'+video_num+'_'+pxl_size+'.json', 'r') as r:
        log_dict = json.load(r)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(3))
    h = int(cap.get(4))
    out = cv2.VideoWriter('./video_'+video_num+'_'+pxl_size+'.mp4', fourcc, 30.0, (w,h))
    start_frame = 12200
    end_frame = 12400
    frame_range = range(start_frame, end_frame)

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % 100 == 0:
            print(frame_id)
        if frame_id > end_frame:
            break
        if frame_id in frame_range:
            if str(frame_id) in log_dict:
                boxes = log_dict[str(frame_id)]
                for box in boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                # 이미지 반전,  0:상하, 1 : 좌우
                out.write(frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
cv2.destroyAllWindows()