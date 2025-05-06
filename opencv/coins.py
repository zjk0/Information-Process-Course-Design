import cv2 as cv
import numpy as np

def create_mask (shape, bool_index):
    if len(shape) > 2:
        shape = (shape[0], shape[1])

    mask = np.zeros(shape, dtype = np.uint8)
    mask[bool_index] = 255

    return mask

# 读取图像
img = cv.imread("./coins.jpg")
img_copy = img.copy()  # 留一个原图像的副本

# 检测图像中的红色区域，也就是背景
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
index_red = (((img_hsv[:, :, 0] >= 0) & (img_hsv[:, :, 0] <= 10)) | ((img_hsv[:, :, 0] >= 160) & (img_hsv[:, :, 0] <= 179))) & \
            ((img_hsv[:, :, 1] >= 170) & (img_hsv[:, :, 1] <= 255)) & \
            ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

index_gold = ((img_hsv[:, :, 0] >= 11) & (img_hsv[:, :, 0] <= 35)) & \
             ((img_hsv[:, :, 1] >= 40) & (img_hsv[:, :, 1] <= 255)) & \
             ((img_hsv[:, :, 2] >= 40) & (img_hsv[:, :, 2] <= 255))

# 得到金色区域掩码
mask_gold = create_mask(img.shape, index_gold)
kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
mask_gold = cv.morphologyEx(mask_gold, cv.MORPH_OPEN, kernel_open)
kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask_gold = cv.morphologyEx(mask_gold, cv.MORPH_CLOSE, kernel_close)

# 得到硬币区域的掩码
img_hsv[index_red, :] = [0, 0, 0]
img_no_red = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
img_gray = cv.cvtColor(img_no_red, cv.COLOR_BGR2GRAY)
img_bin = np.zeros_like(img_gray)
img_bin[img_gray != 0] = 255
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
mask_coin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)  # 使用闭运算合并一些不连续的小的区域

# 连通域检测，并进行圆逼近
contours, _ = cv.findContours(mask_coin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
circle_data = []  # 所有硬币近似圆的数据
for cnt in contours:
    (x, y), r = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    r = int(r)
    circle_data.append(np.array([x, y, r], dtype = int))
    cv.circle(img_copy, center, r, (0, 255, 0), 2)

circles = np.array(circle_data)
cv.imshow("circle", img_copy)

# 金色区域连通域检测，并进行圆逼近
gold_contours, _ = cv.findContours(mask_gold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
gold_circle_data = []  # 金色硬币近似圆的数据
for cnt in gold_contours:
    (x, y), r = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    r = int(r)
    gold_circle_data.append(np.array([x, y, r], dtype = int))

gold_circles = np.array(gold_circle_data)

# 硬币数值提取并进行硬币数值总和计算
value = np.array([0.01, 0.1, 0.02, 0.1, 0.05])
value_index = 0
coin_value = np.zeros(circles.shape[0])
sort_index = np.argsort(circles[:, 2])  # 对第三列进行排序
sort_circles = circles[sort_index]  # 根据第三列的排序结果，重排数组
for i in range(circles.shape[0]):
    x = sort_circles[i, 0]
    y = sort_circles[i, 1]
    if (abs(x - gold_circles[0, 0]) < 10 and abs(y - gold_circles[0, 1]) < 10) or (abs(x - gold_circles[1, 0]) < 10 and abs(y - gold_circles[1, 1]) < 10):
        coin_value[i] = 0.5
    else:
        coin_value[i] = value[int(value_index / 2)]
        value_index += 1

print(f"total: {coin_value.sum() / 2}")

cv.waitKey(0)