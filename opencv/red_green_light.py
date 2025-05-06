import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread("./red_green_light.jpg")
img_copy = img.copy()  # 留一个原图像的副本
cv.imshow("img", img)

# 检测红色和绿色的区域
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转换到hsv空间
index_red = (((img_hsv[:, :, 0] >= 0) & (img_hsv[:, :, 0] <= 10)) | ((img_hsv[:, :, 0] >= 160) & (img_hsv[:, :, 0] <= 179))) & \
            ((img_hsv[:, :, 1] >= 50) & (img_hsv[:, :, 1] <= 255)) & \
            ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

index_green = ((img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85)) & \
              ((img_hsv[:, :, 1] >= 80) & (img_hsv[:, :, 1] <= 255)) & \
              ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

index = index_red | index_green  # 红色或绿色的区域

# 非红绿色区域置为黑色
img_hsv[~index, :] = [0, 0, 0]
img_hsv = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
# cv.imshow("hsv process", img_hsv)

# 制作一个掩码，将红绿色区域提取出来
mask = np.zeros_like(img)
mask[index] = [255, 255, 255]
# cv.imshow("mask", mask)

# 边缘检测
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 原图像转换为灰度图像
# cv.imshow("gray", img_gray)
img_gray = cv.medianBlur(img_gray, 3)  # 进行中值滤波使得图像更加平滑
# cv.imshow("blur", img_gray)
img_edge = cv.Canny(img_gray, 150, 220)  # Canny边缘检测
# cv.imshow("edge", img_edge)

# 将不是灯的边缘去掉
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
mask = cv.dilate(mask, kernel_dilate, iterations = 1)
img_masked = cv.bitwise_and(img_edge, mask)
# cv.imshow("edge masked", img_masked)

# 将灯的边缘转换为一个边缘
kernel_closing = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_masked = cv.morphologyEx(img_masked, cv.MORPH_CLOSE, kernel_closing)
# cv.imshow("edge masked closing", img_masked)

# 连通域检测并且使用圆逼近
img_masked_copy = img_masked.copy()
contours, _ = cv.findContours(img_masked_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
center_list = []
for cnt in contours:
    (x, y), r = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    center_list.append(np.array(center))
    r = int(r)
    cv.circle(img_copy, center, r, (0, 0, 255), 2)

cv.imshow("circle", img_copy)
centers = np.array(center_list)
print(centers)


# 灯亮区域
# index_light = index & ((img_hsv[:, :, 2] >= 230) & (img_hsv[:, :, 2] <= 255))

# mask_light = np.zeros_like(img)
# mask_light[index_light] = [255, 255, 255]
# cv.imshow("mask_light", mask_light)

# img_gray_blur = cv.medianBlur(img_gray, 3)
# mask_light_2 = np.zeros_like(img_gray)
# mask_light_2[img_gray_blur > 200] = 255
# cv.imshow("mask_light_2", mask_light_2)

print("r/R: 红色\ng/G: 绿色\n小写字母表示灯灭, 大写字母表示灯亮")
cv.waitKey(0)