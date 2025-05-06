import cv2 as cv
import numpy as np

img = cv.imread("./coins.jpg")
image = img.copy()
cv.imshow("img", img)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
index_red = (((img_hsv[:, :, 0] >= 0) & (img_hsv[:, :, 0] <= 10)) | ((img_hsv[:, :, 0] >= 160) & (img_hsv[:, :, 0] <= 179))) & \
            ((img_hsv[:, :, 1] >= 170) & (img_hsv[:, :, 1] <= 255)) & \
            ((img_hsv[:, :, 2] >= 50) & (img_hsv[:, :, 2] <= 255))

img_hsv[index_red, :] = [0, 0, 0]
img_no_red = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
cv.imshow("no_red", img_no_red)

img_gray = cv.cvtColor(img_no_red, cv.COLOR_BGR2GRAY)
cv.imshow("img_gray", img_gray)
img_bin = np.zeros_like(img_gray)
img_bin[img_gray != 0] = 255
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
cv.imshow("img_bin", img_bin)

contours, _ = cv.findContours(img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
area_list = []
for cnt in contours:
    area = cv.contourArea(cnt)
    area_list.append(area)

    (x, y), r = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    r = int(r)
    cv.circle(image, center, r, (0, 255, 0), 2)

cv.imshow("circle", image)

areas = np.array(area_list)
areas = areas.reshape(-1, 2)
areas_mean = areas.mean(axis = 1)
areas_mean = np.sort(areas_mean)
print(areas_mean)

coin_value = np.array([0.01, 0.02, 0.05, 0.1, 0.5])

cv.waitKey(0)