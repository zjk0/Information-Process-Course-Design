import cv2 as cv
import numpy as np
from ultralytics import YOLO

# 获取输入视频信息
video = cv.VideoCapture("./input.mp4")
fps = video.get(cv.CAP_PROP_FPS)  # 获取输入视频帧速率
video_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))  # 获取输入视频一帧的宽度
video_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # 获取输入视频一帧的高度
video.release()

# 实例化一个视频写入器
fourcc = cv.VideoWriter.fourcc('m', 'p', '4', 'v')
video_writer = cv.VideoWriter("./output.mp4", fourcc, fps, (video_w, video_h))

# 实例化模型
model = YOLO("yolo11s-pose.pt")

# 采用边推理边输出的形式
results = model.predict(source = "./input.mp4", stream = True)

def get_pos (left, right):
    if left[2] > 0.5 and right[2] > 0.5:
        result = (left[0 : 2] + right[0 : 2]) / 2
    elif left[2] > 0.5 and right[2] <= 0.5:
        result = left[0 : 2]
    elif left[2] <= 0.5 and right[2] > 0.5:
        result = right_ankle[0 : 2]
    else:
        result = np.array([0, 1], dtype = int)

    return result

def get_theta (vector):
    vert = np.array([0, 1], dtype = int)
    cos = np.dot(vector, vert) / (np.linalg.norm(vector) * np.linalg.norm(vert))
    theta = np.arccos(np.clip(cos, -1.0, 1.0))
    return theta

# 遍历每一帧的处理结果
for result in results:
    frame = result.plot()  # 得到一帧处理完成的图片
    person_num = result.boxes.shape[0]  # 得到检测到的人的数量
    if person_num == 0:  # 如果没有检测到人，则不进行后续操作
        video_writer.write(frame)
        continue

    for i in range(person_num):
        # 得到左脚踝和右脚踝的数据，并转换为numpy数组
        left_ankle = result.keypoints.data[i, 15].cpu().numpy()
        right_ankle = result.keypoints.data[i, 16].cpu().numpy()
        ankle = get_pos(left_ankle, right_ankle)
        
        # 得到左膝和右膝的数据，并转换为numpy数组
        left_knee = result.keypoints.data[i, 13].cpu().numpy()
        right_knee = result.keypoints.data[i, 14].cpu().numpy()
        knee = get_pos(left_knee, right_knee)

        # 得到左髋和右髋的数据，并转换为numpy数组
        left_hip = result.keypoints.data[i, 11].cpu().numpy()
        right_hip = result.keypoints.data[i, 12].cpu().numpy()
        hip = get_pos(left_knee, right_knee)

        # 得到左肩和右肩的数据，并转换为numpy数组
        left_shoulder = result.keypoints.data[i, 5].cpu().numpy()
        right_shoulder = result.keypoints.data[i, 6].cpu().numpy()
        shoulder = get_pos(left_shoulder, right_shoulder)

        # 计算膝盖指向脚踝的向量与竖直方向的夹角
        knee_to_ankle = ankle - knee
        theta1 = get_theta(knee_to_ankle)

        # 计算肩部指向髋的向量与竖直方向的夹角
        shoulder_to_hip = hip - shoulder
        theta2 = get_theta(shoulder_to_hip)

        # 如果两个角度都大于45度，则判定为摔倒
        if theta1 > (np.pi / 4) and theta2 > (np.pi / 4):
            box_points = result.boxes.xyxy.cpu().numpy()
            box_points = box_points.astype(int)
            cv.rectangle(frame, (box_points[i, 0], box_points[i, 1]), (box_points[i, 2], box_points[i, 3]), (0, 0, 255), 3)
            cv.putText(frame, "Fall", (box_points[i, 2] + 1, box_points[i, 3]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv.LINE_AA)

    video_writer.write(frame)

video_writer.release()