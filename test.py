#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         utils
# Description:  196像素--3.2cm  平均每厘米61像素
# Author:       Ming_King
# Date:         2022/7/4
# -------------------------------------------------------------------------------
import cv2
import imutils
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
                help="width of the lower-center--most object in the image (in cm)")
args = vars(ap.parse_args())


# 2g-r-b
def CropAndSoilSeg(frame, flag=True):
    # 使用2g-r-b分离土壤与背景
    # 转换为浮点数进行计算
    fsrc = np.array(frame, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)
    kernel = np.ones((3, 3))  # 3*3对角线全1矩阵（内核）
    imgDial = cv2.dilate(bin_img, kernel, iterations=1)  # 膨胀处理
    imgThres = cv2.erode(imgDial, kernel, iterations=1)   # 腐蚀处理
    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(frame)
    color_img = cv2.merge([b8 & imgThres, g8 & imgThres, r8 & imgThres])
    return color_img, imgThres


# ROI区域(根据选取点)选取(非选中区域像素为0)
def roi_mask(img, roi_points):
    # 创建掩膜
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [roi_points], 255)

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# 寻找轮廓
def findcontours_M(image, roi_points):
    # load the image, convert it to grayscale, and blur it slightly
    # image = cv2.imread("img_data\WIN_20220517_S2-F1-N1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    img_roi = roi_mask(edged, roi_points)
    cv2.imshow('img_roi', img_roi)

    # find contours in the edge map
    cnts = cv2.findContours(img_roi.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]

    # # sort the contours from left-to-right and initialize the
    # # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    #
    # # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 20:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

    #     # order the points in the contour such that they appear
    #     # in top-left, top-right, bottom-right, and bottom-left
    #     # order, then draw the outline of the rotated bounding
    #     # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / args["width"]

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)


# 通过不断腐蚀膨胀提取骨架(快速)
def skeleton_extraction_A(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    finished = False
    size = np.size(binary)
    skeleton = np.zeros(binary.shape, np.uint8)
    while not finished:
        eroded = cv2.erode(binary, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        zeros = size - cv2.countNonZero(binary)
        if zeros == size:
            finished = True

    contours, hireachy = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1, 8)
    cv2.imshow("skeleton", image)


#  按特定比例缩放框架 (此方法适合图片,mp4视频和实时视频)
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)  # 宽度比例调整
    height = int(frame.shape[0] * scale)  # 高度比例调整
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    # image = cv2.imread("img_data/WIN_20220517_S2-F1-N1.jpg")
    image = cv2.imread("img_data/17/20220517-S2-F1-N1.jpg")
    img_resized = rescaleFrame(image, 0.3)
    # roi_points = np.array([[210, 450], [210, 375], [1185, 375], [1185, 450]])
    # print(roi_points)
    roi_points = []


    def click_and_crop(event, x, y, flags, param):
        global refPt, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [[x, y]]
            print(x, y)
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append([x, y])
            print(x, y)
            cropping = False

            roi_points = np.array([[refPt[0][0], refPt[1][1]], refPt[0], [refPt[1][0], refPt[0][1]], refPt[1]])
            print(roi_points)
            findcontours_M(img_resized, roi_points=roi_points)


    cv2.namedWindow('17')
    cv2.setMouseCallback('17', click_and_crop)
    cv2.imshow('17', img_resized)
    color_img, imgthres = CropAndSoilSeg(img_resized)
    cv2.imshow('color_img', color_img)
    cv2.imshow('imgthres', imgthres)
    cv2.waitKey(0)

    cv2.destroyAllWindows()