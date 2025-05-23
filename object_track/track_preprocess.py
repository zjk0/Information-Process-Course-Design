from ultralytics import YOLO
import cv2 as cv
import numpy as np

# 使用yolov11的模型
model = YOLO("yolo11s.pt")
results = model.predict(source = "structure.mp4", save = True, classes = [2], stream = True)

# 获取MOT协议格式的特征
features_list = []
feature = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
frame_id = 0

for result in results:
    num = result.boxes.shape[0]

    frame_id += 1
    feature[0] = frame_id

    box_points = result.boxes.xyxy.cpu().numpy()
    score = result.boxes.conf.cpu().numpy()

    for j in range(num):
        feature[2] = box_points[j, 0]
        feature[3] = box_points[j, 1]
        feature[4] = box_points[j, 2] - box_points[j, 0]
        feature[5] = box_points[j, 3] - box_points[j, 1]
        feature[6] = score[j]
        features_list.append(feature.copy())

features = np.array(features_list)
np.savetxt("./yolo_data/data1/det/det.txt", features, fmt = "%d,%d,%f,%f,%f,%f,%f,%d,%d,%d", delimiter = ",")

# 对视频进行抽帧
frame_id = 0
cap = cv.VideoCapture("structure.mp4")
while True:
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    cv.imwrite(f'./yolo_data/data1/img1/{frame_id:06d}.jpg', frame)

cap.release()
