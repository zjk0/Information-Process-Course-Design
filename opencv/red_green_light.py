import cv2 as cv
import numpy as np

def create_mask (shape, bool_index):
    if len(shape) > 2:
        shape = (shape[0], shape[1])

    mask = np.zeros(shape, dtype = np.uint8)
    mask[bool_index] = 255

    return mask

# 读取图像
img = cv.imread("./red_green_light.jpg")
img_copy = img.copy()  # 留一个原图像的副本
cv.imshow("img", img)

# 检测红色和绿色的区域
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转换到hsv空间
index_red = (((img_hsv[:, :, 0] >= 0) & (img_hsv[:, :, 0] <= 10)) | ((img_hsv[:, :, 0] >= 160) & (img_hsv[:, :, 0] <= 179))) & \
            ((img_hsv[:, :, 1] >= 50) & (img_hsv[:, :, 1] <= 255)) & \
            ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

index_green = ((img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 90)) & \
              ((img_hsv[:, :, 1] >= 80) & (img_hsv[:, :, 1] <= 255)) & \
              ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

index = index_red | index_green  # 红色或绿色的区域

# 边缘检测
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 原图像转换为灰度图像
img_gray = cv.medianBlur(img_gray, 3)  # 进行中值滤波使得图像更加平滑
img_edge = cv.Canny(img_gray, 150, 220)  # Canny边缘检测

# 将不是灯的边缘去掉
mask = create_mask(img.shape, index)  # 红绿灯区域的掩码
kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
mask = cv.dilate(mask, kernel_dilate, iterations = 1)
img_masked = cv.bitwise_and(img_edge, mask)

# 将灯的边缘转换为一个边缘，而不是一种类似于环形的结构
kernel_closing = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_masked = cv.morphologyEx(img_masked, cv.MORPH_CLOSE, kernel_closing)

# 连通域检测并且使用圆逼近
img_masked_copy = img_masked.copy()
contours, _ = cv.findContours(img_masked_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 检测最外围的边缘
center_list = []  # 圆心列表
for cnt in contours:
    (x, y), r = cv.minEnclosingCircle(cnt)  #得到近似圆的中心和半径
    center = (int(x), int(y))
    center_list.append(np.array(center))
    r = int(r)
    cv.circle(img_copy, center, r, (0, 0, 255), 2)

centers = np.array(center_list)
cv.imshow("with circle", img_copy)

# 重塑数组
centers = centers[::-1]
centers = centers.reshape(3, -1, 2)
sort_index = np.argsort(centers[:, :, 0], axis = 1)
row_index = np.arange(centers.shape[0])[:, None]
centers = centers[row_index, sort_index, :]
centers = centers.reshape(-1, 2)

# 检测颜色和亮度
lights_color = np.chararray(centers.shape[0])
for i in range(centers.shape[0]):
    row = centers[i, 1]
    column = centers[i, 0]
    h = img_hsv[row, column, 0]
    s = img_hsv[row, column, 1]

    # 处理颜色跳变情况，只对绿色处理
    if i > 0 and i < centers.shape[0] - 1:
        if not ((h >= 0 and h <= 10) or (h >= 160 and h <= 179)):
            h1 = int(img_hsv[centers[i - 1, 1], centers[i - 1, 0], 0])
            h2 = int(img_hsv[centers[i + 1, 1], centers[i + 1, 0], 0])
            if (abs(h - h1) > 30) and (abs(h - h2) > 30):
                h = int((h1 + h2) / 2)

    if h >= 35 and h <= 90:
        if s < 150:
            lights_color[i] = 'G'
        else:
            lights_color[i] = 'g'
    elif (h >= 0 and h <= 10) or (h >= 160 and h <= 179):
        if s < 50:
            lights_color[i] = 'R'
        else:
            lights_color[i] = 'r'
    else:
        lights_color[i] = 'N'

print("r/R: 红色\ng/G: 绿色\n小写字母表示灯灭, 大写字母表示灯亮")
lights_color = np.array(lights_color)
lights_color = lights_color.reshape(3, -1)
print(lights_color)

cv.waitKey(0)