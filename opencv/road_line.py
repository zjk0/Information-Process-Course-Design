import cv2 as cv
import numpy as np

# 读取图像
img_path = "./road_line.jpg"
img = cv.imread(img_path)

# 边缘检测
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_edge = cv.Canny(img_gray, 240, 250)

# 将车道线的区域分离出来
rows = img.shape[0]
columns = img.shape[1]
points = np.array([[0.5 * columns, 0.69 * rows], [0.1 * columns, rows - 1], [0.8 * columns, rows - 1]], dtype = np.int32)
points = [points]  # 定义一个三角形区域的三个顶点
mask = np.zeros((rows, columns), dtype = np.uint8)
mask = cv.fillPoly(mask, points, color = 255)  # 创建一个三角形区域的掩码
img_masked = cv.bitwise_and(img_edge, mask)  # 使用and运算将车道线的区域分离出来

# 霍夫变换检测线段
rho = 1
theta = np.pi / 180
thres = 100
lines = cv.HoughLinesP(img_masked, rho = rho, theta = theta, threshold = thres, minLineLength = 10, maxLineGap = 100)
lines = lines.squeeze(1)

# 得到各条线段的斜率
k_list = []  # 斜率列表
for line in lines:
    k = (line[3] - line[1]) / (line[2] - line[0])
    k_list.append(k)

# 通过斜率来区分是左边车道线还是右边车道线
k_array = np.array(k_list)
lines1 = lines[k_array < 0, :]
lines2 = lines[k_array > 0, :]

# 得到每条车道线中心线段，而不是车道线的边缘
final_lines = np.zeros((2, 4), dtype = int)
final_lines[0, :] = lines1.mean(axis = 0)
final_lines[1, :] = lines2.mean(axis = 0)

# 绘制线段
for line in final_lines:
    cv.line(img, (line[0], line[1]), (line[2], line[3]), color = (0, 0, 255), thickness = 4)

# 显示结果
cv.imshow("line", img)
cv.waitKey(0)