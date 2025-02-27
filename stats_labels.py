#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         stats_labels
# Description:
# Author:       Ming_King
# Date:         2022/10/12
# -------------------------------------------------------------------------------
import xml.dom.minidom
import os
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

classes = ["mazie", "weed_1", "weed_2", "weed_3"]


def test():
    in_file = open('D:/PythonProject/All-Project/yolov5-master/my_datas/VOC2007/Annotations/20220517-S2-F1-N1.xml', encoding='UTF-8')
    out_file = open('D:/PythonProject/All-Project/yolov5-master/my_datas/VOC2007/YOLOLabels/20220517-S2-F1-N1.txt', 'a+', encoding='UTF-8')
    # xmlbox = obj.find('bndbox')
    # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
    #      float(xmlbox.find('ymax').text))
    # bb = convert((w, h), b)
    # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    bb = [1, 1, 1, 1]
    out_file.write(str(1) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()
#
# test()


def txt_stats_labels(root_dir):
    class_name_number = [0, 0, 0, 0]
    # 定义文件夹路径
    folder_path = root_dir

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为 txt 文件
        if filename.endswith(".txt"):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)

            # 打印文件名
            print(f"读取文件：{filename}")

            # 打开文件
            with open(file_path, "r") as file:
                # 逐行读取
                for line in file:
                    # 分割行并提取第一个数字
                    first_number = int(line.split()[0])
                    # 打印第一个数字
                    class_name_number[first_number] += 1
                    print(first_number)

            # 添加空行以分隔不同文件的输出
            print()
    print(class_name_number)


def xml_stats_labels(root_dir):
    doc_xml = os.listdir(root_dir)
    print('num_anno', len(doc_xml))
    classes_list = []
    num_label = {}
    for i in range(0, len(doc_xml)):
        # if i == 23:
        #     break
        path = os.path.join(root_dir, doc_xml[i])
        if os.path.isfile(path):
            # 打开xml文档
            dom = xml.dom.minidom.parse(path)
            # 得到dom元素的label
            root = dom.documentElement
            label = dom.getElementsByTagName('name')
            for j in range(len(label)):
                c1 = label[j]
                class_name = c1.firstChild.data
                # 列表中不存在则存入列表
                if classes_list.count(class_name) == 0:
                    classes_list.append(class_name)
                    num_label[class_name] = 0
                num_label[class_name] += 1
    print('num_classes', len(classes_list))
    print('num_label', num_label)


xml_dir = 'E:/zhiming/result_imgs/crop/7_31_Data/Annotations'
txt_dir = 'F:/Copy-Paste/YOLOLabels'
txt_stats_labels(txt_dir)
# print(len(num_label.keys()))
# print(num_label.values())
# total_labels = []
# for name in classes:
#     i = num_label[name]
#     total_labels.append(i)
# print(sum(total_labels))
# plt.bar(range(len(num_label.keys())), num_label.values(), color='skyblue', tick_label=num_label.keys())
# plt.show()

# classes = ["maize", "seeding1", "seeding2", "seeding#"]
# class_code = range(len(classes))
# print(len(class_code))
# print(class_code[3])
#
# vary = [0, 0]
# vary[0] = 100
# vary[1] = 1
# print(vary)
# for i in range(len(vary)):
#     print(i)
# for i in range(0, 100):
#     img_scale = random.randrange(10, 15, 1)/10
#     print(img_scale)


# def rotate_img(img, angle):
#     h, w = img.shape[:2]
#     rotate_center = (w/2, h/2)
#     # 获取旋转矩阵
#     # 参数1为旋转中心点;
#     # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
#     # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
#     img_scale = random.randrange(10, 15, 1)/10
#     M = cv2.getRotationMatrix2D(rotate_center, angle, img_scale)
#     # 计算图像新边界
#     new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
#     new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
#     # 调整旋转矩阵以考虑平移
#     M[0, 2] += (new_w - w) / 2
#     M[1, 2] += (new_h - h) / 2
#     # if angle in range(-360, 360, 180):
#     #     new_w = w * img_scale
#     #     new_h = h * img_scale
#     # else:
#     #     new_w = h * img_scale
#     #     new_h = w * img_scale
#     rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
#     return rotated_img
#
#
# image = cv2.imread("D:/PythonProject/All-Project/yolov5-master/sci_data/VOC2007/JPEGImages/1.jpg")
# print(image)
# # angle = random.randrange(-360, 360, 90)
# # img = rotate_img(image, angle)
# cv2.imshow("1", image)
# cv2.waitKey()

# num_label = {'maize': 3679, 'weed_1': 3699, 'weed_2:': 3243, 'weed_3':3844, 'weed_4':3421}
# x=[1,2,3,4,5]
# print(num_label.keys())
# x_label = list(num_label.keys())
# print(x_label)
# np.array([list(item.values()) for item in d.values()])
# plt.xticks(x, x_label)#绘制x刻度标签
# plt.bar(x_label, num_label.values(), color='skyblue', tick_label=x_label)
# plt.bar(range(len(num_label.keys())), num_label.values(), color='skyblue', tick_label=num_label.keys())
# plt.show()
