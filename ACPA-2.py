#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         ACPA
# Description:  Adaptive Copy & Paste
# Author:       Ming_King
# Date:         2022/12/25
# -------------------------------------------------------------------------------
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
# classes = [""]
classes = ["maize", "weed_1", "weed_2", "weed_3"]
index_table = []
upper_limit = [50, 3000, 1500, 2500]

data_path = "F:/7_31_Data/"
image_dir = os.path.join(data_path, "JPEGImages/")
# list_imgs = os.listdir(image_dir)  # list image files
labels_dir = os.path.join(data_path, "Annotations/")
out_path = "F:/Copy-Paste"


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
        green_coverage = float(format(green / (w * h), '.2f'))
        return green_coverage


# Image Fusion
def merge_img(M_img, m_img, scale=0.8):
    beta = random.randrange(20, 25, 1) / 100
    merge = cv2.addWeighted(M_img, scale, m_img, beta, gamma=0)
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
    return x, y, w, h


def count_area(labels):
    area = (labels[1] - labels[0]) * (labels[3] - labels[2])
    return area


def label_thresholdX_probabilityA(dir_path, exclude=0, img_size=768):
    file_xml = os.listdir(dir_path)  # list xml files
    class_area = {}
    classes_label = {}
    for a in range(0, len(classes)):
        class_area[str(a)] = 0
        classes_label[str(a)] = 0
    for i in range(0, len(file_xml)):
        # 计算每个种类平均像素面积
        class_label = []
        for j in range(0, len(classes)):
            class_label.append(0)
        file_path = os.path.join(root_dir, file_xml[i])
        in_file = open(file_path, encoding='UTF-8')
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
            class_label[cls_id] += 1
            classes_label[str(cls_id)] += 1
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            class_area[str(cls_id)] += ((b[1]-b[0])*(b[3]-b[2]))
        for cla in range(0, len(classes)):
            if class_label[cla] == 0:
                continue
            class_area[str(cla)] = class_area[str(cla)]/class_label[cla]
    x_ = []
    A = [[1, 2, 3]]
    n = 0
    X = 0
    for i in range(0, len(classes)):
        A.append(classes_label[str(i)])
        if i == exclude:
            continue
        x_.append([classes_label[str(i)], (img_size*img_size)/class_area[str(i)]])
        n += classes_label[str(i)]
    for i in range(0, len(classes_label)):
        if i == exclude:
            continue
        X += (x_[i-1][0]/n*x_[i-1][1])
    B = A.copy()
    for i in range(0, len(classes)):
        p = (max(A[1:])-A[i+1])/A[i+1]
        if p > 2:
            p = 1
        B[i+1] = p
    return int(X), B


# Image Selector Control Unit
def cal_y(labels, green_coverage, X, scale=0.5):
    y1 = sum(labels[1:]) / X
    y2 = green_coverage
    if y2 > 0.8:
        y = 1
    else:
        y = scale * y1 + (1-scale) * y2
    return y


# Get image and labels
def Get_img_Information(file_id):
    img_path = image_dir + '%s.jpg' % file_id
    img = cv2.imread(img_path)
    labels, label_num, class_num = count_object_labels(file_id, if_detail=True)
    return img, labels, label_num, class_num


# Best Index Table
def Index_table(img_dir, Xml_path=None, label_Threshold=20):
    file_img = os.listdir(img_dir)  # list img files
    index = []
    for i in range(0, len(file_img)):
        path_img = os.path.join(img_dir, file_img[i])
        if os.path.isfile(path_img):
            # path_Img = img_dir + file_img[i]
            file_Name = file_img[i]
            (file_Id, file_extention) = os.path.splitext(os.path.basename(file_Name))
            # path_Xml = Xml_path + file_Id + '.xml'
            path_Img = image_dir + file_Id + '.jpg'
            green_coverage = CropAndSoilSeg(path_Img)
            labels, img_labels, _ = count_object_labels(file_Id, if_detail=True)
            labels_num = int(0.4 * img_labels[0] + 0.6 * sum(img_labels[1:]))
            if green_coverage <= 0.3 and labels_num <= label_Threshold:
                index.append([0, file_Id])
    return index


# Whether it can Copy
def AS_CImg(c_img_path):
    green_coverage = CropAndSoilSeg(c_img_path)
    AS_ = True
    if green_coverage >= 0.8:
        AS_ = False
    return green_coverage, AS_


# 3 situations(1; 1,2; 1,2,3;)
def Select_Channel(c_img, c_labels, c_Id, label_num, class_num, green_coverage, Index, X, A, T=0.55):
    p_img = c_img.copy()
    p_labels = []
    target_Id = c_Id
    y = cal_y(label_num, green_coverage, X)
    if y >= T:
        if len(list(set(class_num).intersection(set(A[0])))) != 0:  # here
            ind = random.randint(1, len(Index))
            while Index[ind][0] == 1:
                ind = (ind + 1) % len(Index)
            Id = Index[ind][1]
            target_Id = Id
            Index[ind][0] = 1
            p_img, p_labels, label_num, _ = Get_img_Information(Id)
            p_img, p_labels = copy_paste(p_img, c_img, c_labels, p_labels, A[1:], flag=False)
    else:
        if random.random() <= (1 - y):
            p_labels = c_labels.copy()
            p_img, p_labels = copy_paste(p_img, c_img, c_labels, p_labels, A[1:], flag=True)
        else:
            # to here
            if len(list(set(class_num).intersection(set(A[0])))) != 0:
                img_2_id = random.randint(1, len(list_xml))
                img_2_id = xml_firstname[img_2_id]  # 避免序号不连续
                target_Id = img_2_id
                img_2, labels_2, label_num, _ = Get_img_Information(img_2_id)
                green_c2 = CropAndSoilSeg(image_dir+'%s.jpg' % img_2_id)
                y_2 = cal_y(label_num, green_c2, X)
                if y_2 >= T or random.random() > (1-y_2):
                    ind = random.randint(1, len(Index))
                    while Index[ind][0] == 1:
                        ind = (ind + 1) % len(Index)
                    Id = Index[ind][1]
                    target_Id = Id
                    Index[ind][0] = 1
                    p_img, p_labels, label_num, _ = Get_img_Information(Id)
                    p_img, p_labels = copy_paste(p_img, c_img, c_labels, p_labels, A[1:], flag=True)
                else:
                    p_img = img_2
                    p_labels = labels_2
                    p_img, p_labels = copy_paste(p_img, c_img, c_labels, p_labels, A[1:], flag=True)
    return p_img, p_labels, target_Id


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
    if random.randint(1, 100) < 50:
        rotated_img = cv2.flip(rotated_img, 1)
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


def copy_paste(paste_img, copy_img, copy_labels, paste_labels, A_pro, max_iou=0.0, flag=True):  # probability%
    bh, bw = paste_img.shape[:2]
    angle = random.randrange(-360, 360, 90)
    prob = random.randint(1, 100)/100
    single_image_upper_limit = [10, 25, 15, 20]
    class_upper_num = [1, 3, 1, 3]
    for label in copy_labels:
        num = [1, 1, 1, 1]
        if flag:
            if prob > A_pro[label[0]]:
                continue
        else:
            if prob > (A_pro[label[0]]/num):
                continue
        if upper_limit[label[0]] > 0 or single_image_upper_limit[label[0]] >= 0:
            for j in range(10):
                if num[int(label[0])] > class_upper_num[int(label[0])]:
                    break
                # im4[ymin:ymax, xmin:xmax]
                crop_roi = copy_img[int(label[3]):int(label[4]), int(label[1]):int(label[2])]
                rotate_roi = rotate_img(crop_roi, angle)
                ph, pw = rotate_roi.shape[:2]
                xstart_point = random.randint(0, bw - pw)
                ystart_point = random.randint(0, bh - ph)
                bbox2 = [xstart_point, xstart_point+pw, ystart_point, ystart_point+ph]
                iou_flag = 0
                for label1 in paste_labels:
                    bbox1 = label1[1:]
                    # bbox2 = [xstart_point, xstart_point+pw, ystart_point, ystart_point+ph]
                    iou = cal_iou(bbox1, bbox2)
                    iou_flag = iou
                    if iou > max_iou:
                        break
                if iou_flag > 0:
                    continue
                paste_img[ystart_point:ystart_point+ph, xstart_point:xstart_point+pw] = rotate_roi
                paste_labels.append([label[0], bbox2[0], bbox2[1], bbox2[2], bbox2[3]])
                upper_limit[label[0]] -= 1
                single_image_upper_limit[label[0]] -= 1
                num[int(label[0])] += 1
    return paste_img, paste_labels


def count_object_labels(xml_file_id, if_detail=True):
    in_file = open(data_path + 'Annotations/%s.xml' % xml_file_id, encoding='UTF-8')

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
        if if_detail:
            object_labels.append([cls_id, b[0], b[1], b[2], b[3]])
            label_num[cls_id] += 1
        else:
            label_num[cls_id] += 1
        # bb = convert((w, h), b)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    class_num = []
    for i in range(0, len(classes)):
        if label_num[i] != 0:
            class_num.append(i)
    if if_detail:
        return object_labels, label_num, class_num
    else:
        return label_num


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
            for j in range(len(label)):
                c1 = label[j]
                class_name = c1.firstChild.data
                # 列表中不存在则存入列表
                if classes_list.count(class_name) == 0:
                    classes_list.append(class_name)
                    num_label[class_name] = 0
                num_label[class_name] += 1
    return num_label


if __name__ == "__main__":
    root_dir = 'F:/7_31_Data/Annotations'
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
    X, A = label_thresholdX_probabilityA(root_dir)
    for i in range(0, len(list_xml)):
        path = os.path.join(root_dir, list_xml[i])
        if os.path.isfile(path):
            xml_path = root_dir + list_xml[i]
            voc_path = list_xml[i]
            (voc_Id, voc_extention) = os.path.splitext(os.path.basename(voc_path))
            print(voc_Id)
            image_name = voc_Id + '.jpg'
            image_path = image_dir + image_name
            Green_P, AS = AS_CImg(image_path)
            if not AS:
                continue
            # X, A = label_thresholdX_probabilityA(root_dir) (修)
            Index = Index_table(labels_dir)
            C_img, C_labels, Label_num, Class_num = Get_img_Information(voc_Id)
            Target_img, Target_labels, Target_Id = Select_Channel(C_img, C_labels, voc_Id,
                                                                  Label_num, Class_num, Green_P, Index, X, A)
            xml_2_txt_img(Target_img, Target_labels, Target_Id)
            # a += 1
            # if a == 100:
            #     break

            # 融合图像
            # m_image = CropAndSoilSeg(M_image, if_color=True)
            # merge_image = merge_img(M_image, m_image)
            # labels = count_object_labels(voc_nameWithoutExtention)
            # xml_2_txt_img(merge_image, labels, voc_nameWithoutExtention)

