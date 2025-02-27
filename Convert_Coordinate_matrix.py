#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         Convert_Coordinate_matrix
# Description:
# Author:       Ming_King
# Date:         2023/3/28
# -------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os
import numpy as np
# import random
# import xml.dom.minidom
# import matplotlib.pyplot as plt
# import cv2

classes = ['0', '1', '2', '3']


def get_file_prefix(files):
    file_prefix = []
    fix_files = []
    for file in files:
        filename, suffix = os.path.splitext(file)
        file_prefix.append(int(filename))
    file_prefix = sorted(file_prefix)
    for file in file_prefix:
        fix_files.append(str(file) + suffix)
    return fix_files


def count_object_labels(xml_file_id):
    in_file = open(data_path + '/%s.xml' % xml_file_id, encoding='UTF-8')

    object_labels = []
    label_num = []
    for i in range(0, len(classes)):
        label_num.append(0)

    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        object_labels.append([cls_id, b[0], b[2], b[1], b[3]])
    return object_labels


def Coordinate_Mapping(small_image_boxes, small_image_size, big_image_size, row_idx, col_idx, if_midpoints=False):
    mapped_boxes = []

    row_step = 768
    col_step = 768
    if if_midpoints:
        for cls, mid in small_image_boxes:
            # 计算小图在大图上的偏移量
            x_offset = col_idx * col_step
            y_offset = row_idx * row_step

            # 将小图上的样本框映射回大图
            mapped_x = mid[0] + x_offset
            mapped_y = mid[1] + y_offset
            mapped_box = (cls, (mapped_x, mapped_y))
            mapped_boxes.append(mapped_box)
        return mapped_boxes
    else:
        for cls, xmin, ymin, xmax, ymax in small_image_boxes:
            # xmin, ymin, xmax, ymax = box

            # 计算小图在大图上的偏移量
            x_offset = col_idx * col_step
            y_offset = row_idx * row_step

            # 将小图上的样本框映射回大图
            mapped_xmin = xmin + x_offset
            mapped_ymin = ymin + y_offset
            mapped_xmax = xmax + x_offset
            mapped_ymax = ymax + y_offset
            mapped_box = (cls, mapped_xmin, mapped_ymin, mapped_xmax, mapped_ymax)
            mapped_boxes.append(mapped_box)

        return mapped_boxes


def DIY_nms(detections, iou_threshold=0, small_box_threshold=0.6):
    if len(detections) == 0:
        return []

    classes = np.array([det[0] for det in detections])
    boxes = np.array([det[1:] for det in detections], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    unique_classes = np.unique(classes)
    keep = []

    for label in unique_classes:
        label_indices = np.where(classes == label)[0]
        label_boxes = boxes[label_indices]
        label_areas = areas[label_indices]

        order = label_areas.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep.append(label_indices[i])

            xx1 = np.maximum(x1[label_indices[i]], x1[label_indices[order[1:]]])
            yy1 = np.maximum(y1[label_indices[i]], y1[label_indices[order[1:]]])
            xx2 = np.minimum(x2[label_indices[i]], x2[label_indices[order[1:]]])
            yy2 = np.minimum(y2[label_indices[i]], y2[label_indices[order[1:]]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            # 计算重叠区域占小边界框总面积的比例
            overlap_ratio = intersection / np.minimum(label_areas[i], label_areas[order[1:]])

            inds = np.where(overlap_ratio <= small_box_threshold)[0]
            order = order[inds + 1]

    return keep


if __name__ == '__main__':
    data_path = 'F:/Test_Images/Annotations'
    dirs = os.listdir(data_path)  # 读取所有的文件
    # dirs = sorted(dirs)  # 文件字符串排序（supply）
    # Sort the list of files by file prefix
    dirs = get_file_prefix(dirs)
    for file in dirs:
        file_pname = os.path.splitext(file)[0]
        ground_truth_labels = count_object_labels(file_pname)
        print(ground_truth_labels)
