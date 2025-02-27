#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         Tool_Kits
# Description:
# Author:       Ming_King
# Date:         2022/12/27
# -------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os
import numpy as np
import random
import xml.dom.minidom
# import matplotlib.pyplot as plt
import cv2


# 2g-r-b
def CropAndSoilSeg(img_path, if_color=False):
    # 使用2g-r-b分离土壤与背景
    # 转换为浮点数进行计算
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    fsrc = np.array(img, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # (thresh, bin_img) = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)
    kernel = np.ones((3, 3))  # 3*3对角线全1矩阵（内核）
    imgDial = cv2.dilate(bin_img, kernel, iterations=1)  # 膨胀处理
    imgThres = cv2.erode(imgDial, kernel, iterations=1)  # 腐蚀处理
    # 得到彩色的图像
    if if_color:
        (b8, g8, r8) = cv2.split(img)
        color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
        return color_img
    else:
        green = cv2.countNonZero(imgThres)
        # probability = float(format(100 - (green / (w * h) * 100), '.1f'))
        green_coverage = float(format((green / (w * h) * 100), '.1f'))
        return green_coverage


# Image Fusion
def merge_img(M_img, m_img, scale=0.8):
    beta = random.randrange(20, 25, 1) / 100
    merge = cv2.addWeighted(M_img, scale, m_img, beta, gamma=0)
    return merge


# [x1,y1,x2,y2] area
def count_area(labels):
    area = (labels[1] - labels[0]) * (labels[3] - labels[2])
    return area


# Image Selector Control Unit
def cal_y(labels, green_coverage, X, scale=0.5):
    y1 = sum(labels[1:]) / X
    y2 = green_coverage
    if y2 > 0.8:
        y = 1
    else:
        y = scale * y1 + (1-scale) * y2
    return y


# Iou
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2)
    # if xmin > xmax or ymin > ymax:
    #     iou = 0
    # else:
    #     iou = 1
    return iou