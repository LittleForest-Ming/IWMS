#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         Merge
# Description:
# Author:       Ming_King
# Date:         2023/3/2
# -------------------------------------------------------------------------------

def merge_iou_boxes(boxes, iou_thresh):
    """
    Merges overlapping rectangular boxes using Intersection over Union (IoU) metric.

    Args:
    boxes (list): a list of tuples containing top-left and bottom-right pixel points of rectangular boxes.
    iou_thresh (float): the threshold above which the boxes will be merged.

    Returns:
    merged_boxes (list): a list of merged boxes.
    """
    merged_boxes = []

    # Sort the boxes in the order of their x-coordinates
    # boxes = sorted(boxes, key=lambda x: x[0][0])

    # Loop through each box and merge with the next box if their IoU is above the threshold
    i = 0
    while i < len(boxes):
        box = boxes[i]

        for j in range(i + 1, len(boxes)):
            iou = calc_iou(box, boxes[j])

            if iou > iou_thresh:
                box = merge_boxes(box, boxes[j])

        merged_boxes.append(box)
        i += 1

    indexs = []
    merged_boxes_all = []
    for index in range(0, len(merged_boxes)):
        for i in range(0, len(boxes)):
            if merged_boxes[index] == boxes[i]:
                indexs.append(index)

    for i in range(0, len(merged_boxes)):
        if i in indexs:
            continue
        merged_boxes_all.append(merged_boxes[i])

    return merged_boxes_all


def calc_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two boxes.

    Args:
    box1 (tuple): a tuple containing top-left and bottom-right pixel points of the first box.
    box2 (tuple): a tuple containing top-left and bottom-right pixel points of the second box.

    Returns:
    iou (float): the IoU of the two boxes.

    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2)
    return iou


def merge_boxes(box1, box2):
    """
    Merges two boxes into one.

    Args:
    box1 (tuple): a tuple containing top-left and bottom-right pixel points of the first box.
    box2 (tuple): a tuple containing top-left and bottom-right pixel points of the second box.

    Returns:
    merged_box (tuple): a tuple containing top-left and bottom-right pixel points of the merged box.
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    merged_box = (x1, y1, x2, y2)

    return merged_box


# Sample data
boxes = [(100, 100, 200, 200),
         (100, 100, 250, 250),
         (300, 300, 400, 400),
         (320, 320, 420, 420),
         (50, 50, 150, 150)]

# Merge boxes with IoU threshold of 0.5
merged_boxes = set(merge_iou_boxes(boxes, 0.2))


# Print the merged boxes
for box in merged_boxes:
    print(box)

