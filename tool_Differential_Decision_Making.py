#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         tool-Differenttial-Decision-Making
# Description:
# Author:       Ming_King
# Date:         2023/3/27
# -------------------------------------------------------------------------------
import cv2
import matplotlib
import numpy as np
from imutils import contours, perspective
import imutils
import matplotlib.pyplot as plt
#  [(119, 172, 48), (180, 105, 255), (76, 175, 80), (0, 119, 255)]
color_map = [(119, 172, 48), (180, 105, 255), (76, 175, 80), (0, 119, 255), (0, 0, 255), (188, 18, 162),
             (42, 42, 165), (128, 128, 128)]
weed_pixs_color_map = [(119, 172, 48), (180, 105, 255), (76, 175, 80), (0, 119, 255), (0, 0, 255), (188, 18, 162),
                       (42, 42, 165), (128, 128, 128)]
class_name = ["0", "1", "2", "3"]


# 求中心点
def midpoint(leftx, lefty, w, h):
    return leftx + w / 2, lefty + h / 2


# 2g-r-b
def CropAndSoilSeg(frame, if_color=False):
    # 使用2g-r-b分离土壤与背景
    # 转换为浮点数进行计算
    fsrc = np.array(frame, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)
    kernel = np.ones((3, 3))  # 5*5对角线全1矩阵（内核）
    imgDial = cv2.dilate(bin_img, kernel, iterations=1)  # 膨胀处理
    imgThres = cv2.erode(imgDial, kernel, iterations=1)  # 腐蚀处理
    # 得到彩色的图像
    if if_color:
        (b8, g8, r8) = cv2.split(frame)
        color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
        return color_img, imgThres
    else:
        return imgThres


# 寻找轮廓
def findcontours_M(image, binary, if_draw=True):
    # load the image, convert it to grayscale, and blur it slightly
    # image = cv2.imread("img_data\WIN_20220517_S2-F1-N1.jpg")
    # binary = CropAndSoilSeg(frame=image, if_color=if_color)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(binary, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=2)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]

    # # sort the contours from left-to-right and initialize the
    # # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    #
    # # loop over the contours individually
    weed_midpoints = []  # store weed_central_points
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 30:
            continue

        # compute the rotated bounding box of the contour
        ###################################################
        # box = cv2.minAreaRect(c)
        # box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        # box = np.array(box, dtype="int")
        box = cv2.boundingRect(c)
        weed_mid = xywh2xy_center(box[0], box[1], box[2], box[3])
        weed_midpoints.append((4, weed_mid))

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        # box = perspective.order_points(box)
        if if_draw:
            ####################################################
            # cv2.drawContours(image, [box.astype("int")], -1, (50, 50, 128), 2)
            # cv2.putText(image, text='Weed', org=(box[0][0], box[0][1]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.6,
            #             color=(50, 50, 128),
            #             thickness=2,
            #             lineType=cv2.LINE_AA)
            cv2.rectangle(image, (box[0], box[1] + box[3]), (box[0] + box[2], box[1]),
                          (128, 128, 128), 2, lineType=cv2.LINE_AA)
            cv2.putText(image, text='4', org=(box[0], box[1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(128, 128, 128),
                        thickness=2,
                        lineType=cv2.LINE_AA)
    return image, weed_midpoints


# 去除苗或杂草
def separate_2dif(image, target_ayy, if_draw=True, if_weedpixs=False, if_differentiation=True, green_up_limit=0.09):
    binary = CropAndSoilSeg(image)
    binary2 = binary.copy()
    if_Warning_Area = False
    weed_pixs = 0
    # cv2.imshow('bin', binary)
    # cv2.waitKey(0)
    maize_midpoints = []
    weed_midpoints_1 = []
    for (x, y, w, h, cls) in target_ayy:
        binary[y-6:(y+h+6), x-6:(x+w+6)] = 0
        if class_name[int(cls)] == '0':
            binary2[y:y + h, x:x + w] = 0
        mid = xywh2xy_center(x, y, w, h)
        if class_name[int(cls)] == '0':
            maize_midpoints.append(mid)
        else:
            weed_midpoints_1.append((cls, mid))
        x_min, y_min, x_max, y_max = xywh2xyxy(x, y, w, h)
        cv2.putText(img=image,
                    text=class_name[int(cls)],
                    org=(int(x_min), int(y_min) - 5),
                    fontFace=0,
                    fontScale=0.6,
                    color=color_map[int(cls)],
                    thickness=2)
        cv2.rectangle(img=image,
                      pt1=(int(x_min), int(y_min)),
                      pt2=(int(x_max), int(y_max)),
                      color=color_map[int(cls)],
                      thickness=2,
                      lineType=cv2.LINE_AA)
    if if_draw:
        # if if_differentiation:
        #     green_percentage = cv2.countNonZero(binary)
        #     green_percentage = green_percentage / (binary.shape[0] * binary.shape[1])
        #     if green_percentage >= green_up_limit:
        #         if_Warning_Area = True
        #     else:
        if if_differentiation:
            img, weed_midpoints = findcontours_M(image, binary.copy(), if_draw=False)
        else:
            img, weed_midpoints = findcontours_M(image, binary.copy(), if_draw=if_draw)

        weed_midpoints_1.extend(weed_midpoints)
        if if_weedpixs:
            green_percentage = cv2.countNonZero(binary2)  # 计算二值化图像非零像素数量
            weed_pixs = green_percentage / (binary2.shape[0] * binary2.shape[1])
            if weed_pixs >= green_up_limit:
                if_Warning_Area = True
            return img, weed_midpoints_1, maize_midpoints, weed_pixs, if_Warning_Area
        return img, weed_midpoints_1, maize_midpoints, weed_pixs, if_Warning_Area


# xywh -> x_min,y_min,x_max,y_max
def xywh2xyxy(x, y, w, h):
    return x, y, x + w, y + h


# 计算中心点
def xywh2xy_center(x, y, w, h):
    return x + w / 2, y + h / 2


#  按特定比例缩放框架 (此方法适合图片,mp4视频和实时视频)
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)  # 宽度比例调整
    height = int(frame.shape[0] * scale)  # 高度比例调整
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# 图像拼接
def image_mosaic(big_img, area_img, area_axis, scale=0.1, if_scale=False, if_rgb=True):
    if if_rgb:
        big_img[area_axis[1]:(area_axis[1] + area_axis[2]),
                area_axis[0]:(area_axis[0] + area_axis[2])] = area_img  # (H,W)
        if if_scale:
            big_resized_img = rescaleFrame(big_img, scale=scale)
            return big_resized_img
        return big_img
    else:
        big_img[area_axis[1]:area_axis[1] + area_axis[2], area_axis[0]:area_axis[0] + area_axis[2]] = area_img
        if if_scale:
            big_resized_img = rescaleFrame(big_img, scale=scale)
            return big_resized_img
        return big_img


# 创建图片模板
def create_img(width, height, if_rgb=True, dtype="uint8"):
    if if_rgb:
        img = np.zeros((height, width, 3), dtype)
    else:
        img = np.zeros((height, width), dtype)
    return img


# 物体中心点绘制
def midpoints_plot(image, seeding_midpoints_arr, weed_midpoints_arr, if_weed=True, if_seeding=True):
    mask = np.ones((image.shape[1], image.shape[0], 3), dtype=np.uint8)
    mask *= 255  # white background
    if if_weed:
        for (cls, loc) in weed_midpoints_arr:  # 绘制草点
            cv2.circle(mask, (int(loc[0]), int(loc[1])), 6, color=weed_pixs_color_map[cls], thickness=-1)
    if if_seeding:
        for (x, y) in seeding_midpoints_arr:  # 绘制作物点
            cv2.circle(mask, (int(x), int(y)), 6, color=(255, 0, 0), thickness=-1)
    return mask


# 绘制等高线图
def plt_contourf(x_weight, y_height, z_arr, if_trans=False):
    x_value = np.arange(0, x_weight, 1)
    y_value = np.arange(0, y_height, 1)
    if if_trans:
        z_arr = np.array(z_arr).reshape(x_weight, y_height)

    # 转换成矩阵数据
    x_arr, y_arr = np.meshgrid(x_value, y_value)

    # 填充等高线
    fig, ax = plt.subplots()
    ax.contourf(x_arr, y_arr, z_arr, cmap='hot', levels=np.linspace(0, 255, 255), extend='both')
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    ax.set_title('contourf')

    # 保存图表
    plt.savefig('table1.png')


# 图像拼接展示
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])   # 水平拼接
        ver = np.vstack(hor)   # 垂直拼接
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
