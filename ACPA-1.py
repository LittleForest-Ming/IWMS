#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         my_labels
# Description:
# Author:       Ming_King
# Date:         2022/10/12
# -------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os
import numpy as np
import random
import xml.dom.minidom
# import matplotlib.pyplot as plt
import cv2

# 标签名称
classes = ["mazie", "typeone_weed", "typetwo_weed", "pb_area"]

# wd = os.getcwd()
# base_dir = os.path.join(wd, "my_datas/")
data_path = "D:/PythonProject/All-Project/yolov5-master/sci_data/VOC2007/"
image_dir = os.path.join(data_path, "JPEGImages/")
# list_imgs = os.listdir(image_dir)  # list image files
out_path = "D:/PythonProject/All-Project/yolov5-master/sci_data/VOC2007/test/copy_paste"


# 2g-r-b
def CropAndSoilSeg(img, if_color=False):
    # 使用2g-r-b分离土壤与背景
    # 转换为浮点数进行计算
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
    kernel = np.ones((3, 3))  # 5*5对角线全1矩阵（内核）
    imgDial = cv2.dilate(bin_img, kernel, iterations=1)  # 膨胀处理
    imgThres = cv2.erode(imgDial, kernel, iterations=1)  # 腐蚀处理
    # 得到彩色的图像
    if if_color:
        (b8, g8, r8) = cv2.split(img)
        color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
        return color_img
    else:
        green = cv2.countNonZero(imgThres)
        probability = float(format(100 - (green / (w * h) * 100), '.1f'))
        return probability


def merge_img(M_img, m_img):
    beta = random.randrange(20, 25, 1) / 100
    merge = cv2.addWeighted(M_img, 0.8, m_img, beta, gamma=0)
    return merge


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def count_area(labels):
    area = (labels[1] - labels[0]) * (labels[3] - labels[2])
    return area


def increase_probability(total_labels, green_percentage, label_num):
    vary_probability = []
    for i in classes:
        index = int(total_labels[i])
        pos = float(format((1 - index / label_num) * green_percentage, '0.1'))
        vary_probability.append(pos)
    return vary_probability


def increase_ratio(vary_labels, balance_Point=10):
    class_code = range(len(classes))
    vary_area = []
    sum_labels = []
    increase_nums = []
    for i in range(len(classes)):
        vary_area.append(0)
        sum_labels.append(0)
        increase_nums.append(0)
    for label in vary_labels:
        for i in class_code:
            if label[0] == class_code[i]:
                vary_area[i] += count_area(label[1:])
                sum_labels[i] += 1
    max_area_object = [max(vary_area), vary_area.index(max(vary_area))]
    for i in range(len(classes)):
        if i == max_area_object[1]:
            if sum_labels[i] < balance_Point:
                increase_nums[i] += round(np.random.rand() * 3)
            continue
        if vary_area[i] == 0:
            increase_nums[i] = round((max_area_object[0] - vary_area[i]) / pow(balance_Point, 2)) + 3
        else:
            increase_nums[i] = round((max_area_object[0] - vary_area[i]) /
                                     (vary_area[i] * balance_Point) * sum_labels[i]) + 3
    return increase_nums


def rotate_img(img, angle):
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    # 获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    img_scale = random.randrange(10, 15, 1)/10
    M = cv2.getRotationMatrix2D(rotate_center, angle, img_scale)
    # 计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    # if angle in range(-360, 360, 180):
    #     new_w = w * img_scale
    #     new_h = h * img_scale
    # else:
    #     new_w = h * img_scale
    #     new_h = w * img_scale
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img


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


def copy_paste(target_img, paste_img, paste_labels, target_labels, upper_limit, probability, max_iou=0.0):  # probability%
    bh, bw = target_img.shape[:2]
    angle = random.randrange(-360, 360, 90)
    prob = random.randint(1, 100)/100
    for label in paste_labels:
        if prob > probability[label[0]]:
            break
        if upper_limit[label[0]] > 0:
            flag = 0
            for j in range(10):
                if flag >= 3:
                    break
                crop_roi = paste_img[int(label[3]):int(label[4]), int(label[1]):int(label[2])]  # im4[ymin:ymax, xmin:xmax]
                rotate_roi = rotate_img(crop_roi, angle)
                ph, pw = rotate_roi.shape[:2]
                xstart_point = random.randint(0, bw - pw)
                ystart_point = random.randint(0, bh - ph)
                bbox2 = [xstart_point, xstart_point+pw, ystart_point, ystart_point+ph]
                iou_flag = 0
                for label1 in target_labels:
                    bbox1 = label1[1:]
                    # bbox2 = [xstart_point, xstart_point+pw, ystart_point, ystart_point+ph]
                    iou = cal_iou(bbox1, bbox2)
                    iou_flag = iou
                    if iou > max_iou:
                        break
                if iou_flag > 0:
                    continue
                target_img[ystart_point:ystart_point+ph, xstart_point:xstart_point+pw] = rotate_roi
                target_labels.append([label[0], bbox2[0], bbox2[1], bbox2[2], bbox2[3]])
                upper_limit[label[0]] -= 1
                flag += 1
    return target_img, target_labels


def count_object_labels(xml_file_id):
    in_file = open(data_path + '/Annotations/%s.xml' % xml_file_id, encoding='UTF-8')
    object_labels = []
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
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
        object_labels.append([cls_id, b[0], b[1], b[2], b[3]])
        # bb = convert((w, h), b)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()

    return object_labels


def xml_2_txt_img(change_img, labels, image_id):
    out_file = open(out_path + '/YOLOLabels/%s.txt' % image_id, 'w', encoding='UTF-8')
    h, w = change_img.shape[:2]
    for i in range(0, len(labels)):
        b = labels[i][1:]
        bb = convert((w, h), b)
        out_file.write(str(labels[i][0]) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
    cv2.imwrite(out_path + '/JPEGImages/%s.jpg' % image_id, change_img)


def convert_annotations(image_id):
    in_file = open('my_datas/VOC2007/Annotations/%s.xml' % image_id, encoding='UTF-8')
    out_file = open('my_datas/VOC2007/YOLOLabels/%s.txt' % image_id, 'a+', encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        # xmlbox = obj.find('bndbox')
        # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
        #      float(xmlbox.find('ymax').text))
        # bb = convert((w, h), b)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


def Stats_labels(rootdir):
    doc_xml = os.listdir(rootdir)
    print('num_anno', len(doc_xml))
    classes_list = []
    num_label = {}
    for i in range(0, len(doc_xml)):
        path = os.path.join(rootdir, doc_xml[i])
        if os.path.isfile(path):
            # 打开xml文档
            dom = xml.dom.minidom.parse(path)
            # 得到dom元素的label
            root = dom.documentElement
            label = dom.getElementsByTagName('name')
            for i in range(len(label)):
                c1 = label[i]
                class_name = c1.firstChild.data
                # 列表中不存在则存入列表
                if classes_list.count(class_name) == 0:
                    classes_list.append(class_name)
                    num_label[class_name] = 0
                num_label[class_name] += 1
    # print('num_classes', len(classes_list))
    # print('num_label', num_label)
    # plt.bar(range(len(num_label.keys())), num_label.values(), color='skyblue', tick_label=num_label.keys())
    # plt.show()
    # total_labels = []
    # for name in classes:
    #     i = int(num_label[name])
    #     total_labels.append(i)
    return num_label


if __name__ == "__main__":
    root_dir = 'D:/PythonProject/All-Project/yolov5-master/sci_data/VOC2007/Annotations'
    list_xml = os.listdir(root_dir)  # list image files
    # print(list_xml)

    # 获取所有xml文件前缀
    xml_firstname = []
    for i in range(0, len(list_xml)):
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(list_xml[i]))
        xml_firstname.append(voc_nameWithoutExtention)

    name_labels = Stats_labels(root_dir)
    data = (i for i in name_labels.values())
    all_labels_num = sum(data)
    a = 0
    for i in range(0, len(list_xml)):
        path = os.path.join(root_dir, list_xml[i])
        if os.path.isfile(path):
            xml_path = root_dir + list_xml[i]
            voc_path = list_xml[i]
            # (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(root_dir))
            (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
            print(voc_nameWithoutExtention)
            image_name = voc_nameWithoutExtention + '.jpg'
            image_path = image_dir + image_name
            M_image = cv2.imread(image_path)
            if random.randint(1, 100) < 80:
                all_labels = count_object_labels(voc_nameWithoutExtention)
                vary_increase_num = increase_ratio(all_labels)
                green_probability = CropAndSoilSeg(M_image)
                probability = increase_probability(name_labels, green_probability/100, all_labels_num)
                target_image, target_labels = copy_paste(M_image, M_image, all_labels,
                                                         all_labels, vary_increase_num, probability)
                xml_2_txt_img(target_image, target_labels, voc_nameWithoutExtention)
            else:
                random_Id = random.randint(1, len(list_xml))
                random_imdId = xml_firstname[random_Id]
                target_img_path = image_dir + random_imdId + '.jpg'
                all_labels_paste = count_object_labels(voc_nameWithoutExtention)
                all_labels_target = count_object_labels(random_imdId)
                target_img = cv2.imread(target_img_path)
                vary_increase_num = increase_ratio(all_labels_target)
                green_probability = CropAndSoilSeg(target_img)
                probability = increase_probability(name_labels, green_probability / 100, all_labels_num)
                target_image, target_labels = copy_paste(target_img, M_image, all_labels_paste,
                                                         all_labels_target, vary_increase_num, probability)
                xml_2_txt_img(target_image, target_labels, str(random_imdId))
            # a += 1
            # if a == 100:
            #     break

            # 融合图像
            # m_image = CropAndSoilSeg(M_image, if_color=True)
            # merge_image = merge_img(M_image, m_image)
            # labels = count_object_labels(voc_nameWithoutExtention)
            # xml_2_txt_img(merge_image, labels, voc_nameWithoutExtention)

