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


def merge_boxes_1(box1, box2):
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


# better
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_b - x1_b) * (y2_b - y1_b)

    union = area_box1 + area_box2 - intersection

    return intersection / union


def merge_boxes(detections, iou_threshold=0.5):
    results = []
    detections = sorted(detections, key=lambda x: x[0])

    while len(detections) > 0:
        current_class, current_box = detections.pop(0)
        if current_class == 0:
            continue

        to_merge = [current_box]
        for i, (other_class, other_box) in enumerate(detections):
            if current_class == other_class and compute_iou(current_box, other_box) > iou_threshold:
                to_merge.append(other_box)
                del detections[i]

        merged_box = (
            min(box[0] for box in to_merge),
            min(box[1] for box in to_merge),
            max(box[2] for box in to_merge),
            max(box[3] for box in to_merge),
        )
        results.append((current_class, *merged_box))

    return results


if __name__ == '__main__':
    # A simple test
    detections = [
        (1, 10, 10, 50, 50),
        (1, 20, 20, 60, 60),
        (2, 70, 70, 110, 110),
        (2, 80, 80, 120, 120),
        (3, 130, 130, 170, 170),
        (0, 140, 140, 180, 180),
    ]

    merged_detections = merge_boxes(detections, iou_threshold=0.45)
    print(merged_detections)


