#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         crop
# Description:
# Author:       Ming_King
# Date:         2022/7/10
# -------------------------------------------------------------------------------
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal_array
from osgeo import gdal

Image.MAX_IMAGE_PIXELS = None

# ##############################读取图片信息###################################
# ##图片路径设置
# in_ds = gdal.Open("F:/Date/东北黑土苗草数据/result_imgs/333.tif")
# out_path = "D:/PythonProject/yolov5/crop_data_8"
#
# # 读取原图中的每个波段，通道数从1开始，默认前三波段
# in_band1 = in_ds.GetRasterBand(1)
# in_band2 = in_ds.GetRasterBand(2)
# in_band3 = in_ds.GetRasterBand(3)
#
# # 获取原图的原点坐标信息
# ori_transform = in_ds.GetGeoTransform()
# top_left_x = ori_transform[0]  # 左上角x坐标
# top_left_y = ori_transform[3]  # 左上角y坐标
# w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
# n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
#
# # 读取原图中的每个波段，通道数从1开始，默认前三波段
# in_band1 = in_ds.GetRasterBand(1)
# in_band2 = in_ds.GetRasterBand(2)
# in_band3 = in_ds.GetRasterBand(3)
#
# # 获取原图的原点坐标信息
# ori_transform = in_ds.GetGeoTransform()
# top_left_x = ori_transform[0]  # 左上角x坐标
# top_left_y = ori_transform[3]  # 左上角y坐标
# w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
# n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

# 要裁剪图像的大小
img_w = 256*5
img_h = 256*5
crop = [img_w, img_h]
cropsize = 400
cropsize_w = 91
cropsize_h = 37


# 读取路径下图片的名称
def file_name(dir_path):
    dirs = os.listdir(dir_path)  # 读取所有的文件
    L = []
    for file in dirs:
        if os.path.splitext(file)[1] == '.tif':  # 只读取固定后缀的文件
            image_name = str(dir_path + '/' + file)
            L.append(image_name)
    return L


# image_sets = file_name('D:/Data Set/东农')  # 图片存贮路径
image_sets = ['E:/2023070720m.tif']


# 添加噪声
def add_noise(img):
    drawObject = ImageDraw.Draw(img)
    for i in range(250):  # 添加点噪声
        temp_x = np.random.randint(0, img.size[0])
        temp_y = np.random.randint(0, img.size[1])
        drawObject.point((temp_x, temp_y), fill="white")  # 添加白色噪声点,噪声点颜色可变
    return img


# 色调增强
def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img


def data_augment(src_roi, label_roi):
    # 图像和标签同时进行90，180，270旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(90)
        label_roi = label_roi.rotate(90)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(180)
        label_roi = label_roi.rotate(180)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(270)
        label_roi = label_roi.rotate(270)
    # 图像和标签同时进行竖直旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_LEFT_RIGHT)
        label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)
    # 图像和标签同时进行水平旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_TOP_BOTTOM)
        label_roi = label_roi.transpose(Image.FLIP_TOP_BOTTOM)
    # 图像进行高斯模糊
    if np.random.random() < 0.25:
        src_roi = src_roi.filter(ImageFilter.GaussianBlur)
    # 图像进行色调增强
    if np.random.random() < 0.25:
        src_roi = random_color(src_roi)
    # 图像加入噪声
    if np.random.random() < 0.2:
        src_roi = add_noise(src_roi)
    return src_roi, label_roi


def crop_img(img, cropsize, overlap):
    """
    裁剪图像为指定格式并保存成tiff
    输入为array形式的数组
    """
    num = 0
    height = img.shape[1]
    width = img.shape[2]
    print(height)
    print(width)

    # 从左上开始裁剪
    for i in range(int(height / (cropsize * (1 - overlap)))):  # 行裁剪次数
        for j in range(int(width / (cropsize * (1 - overlap)))):  # 列裁剪次数
            cropped = img[:,  # 通道不裁剪
                      int(i * cropsize * (1 - overlap)): int(i * cropsize * (1 - overlap) + cropsize),
                      int(j * cropsize * (1 - overlap)): int(j * cropsize * (1 - overlap) + cropsize),
                      ]  # max函数是为了防止i，j为0时索引为负数

            num = num + 1
            target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
            gdal_array.SaveArray(cropped, target, format="GTiff")

    #  向前裁剪最后的列
    for i in range(int(height / (cropsize * (1 - overlap)))):
        cropped = img[:,  # 通道不裁剪
                  int(i * cropsize * (1 - overlap)): int(i * cropsize * (1 - overlap) + cropsize),  # 所有行
                  width - cropsize: width,  # 最后256列
                  ]

        num = num + 1
        target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
        gdal_array.SaveArray(cropped, target, format="GTiff")

    # 向前裁剪最后的行
    for j in range(int(width / (cropsize * (1 - overlap)))):
        cropped = img[:,  # 通道不裁剪
                  height - cropsize: height,  # 最后256行
                  int(j * cropsize * (1 - overlap)): int(j * cropsize * (1 - overlap) + cropsize),  # 所有列
                  ]

        num = num + 1
        target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
        gdal_array.SaveArray(cropped, target, format="GTiff")


    # 裁剪右下角
    cropped = img[:,  # 通道不裁剪
              height - cropsize: height,
              width - cropsize: width,
              ]

    num = num + 1
    target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
    gdal_array.SaveArray(cropped, target, format="GTiff")


# 滑动创建区域数据集
def creat_dataset(overlap=0.00):
    print('creating dataset...')
    for i in tqdm(range(len(image_sets))):
        num = 0
        src_img = Image.open(image_sets[i])  # 3 channels
        width, height = src_img.size[:2]
        arrSlope = []  # 用于存储每个小区域的（X, Y, W=H）坐标
        ranks_area = [[int(height / (cropsize_h * (1 - overlap))),
                       int(width / (cropsize_w * (1 - overlap)))]]  # 用于存取每个区域列数行数 [0]存储总行数列数

        # 对图像进行裁剪，这里大小为1024*1024
        # 从左上开始裁剪
        for i in range(int((height / (cropsize_h * (1 - overlap))))):  # 行裁剪次数
            for j in range(int((width / (cropsize_w * (1 - overlap))))):  # 列裁剪次数
                # max函数是为了防止i，j为0时索引为负数
                cropped = src_img.crop((int(j * cropsize_w * (1 - overlap)), int(i * cropsize_h * (1 - overlap)),
                                        int(j * cropsize_w * (1 - overlap) + cropsize_w),
                                        int(i * cropsize_h * (1 - overlap) + cropsize_h)))
                num = num + 1
                arrSlope.append([int(j * cropsize_w * (1 - overlap)), int(i * cropsize_h * (1 - overlap)), cropsize_w,
                                 cropsize_h])
                ranks_area.append([i, j])
                #########
                # out_band1 = in_band1.ReadAsArray(int(j * cropsize * (1 - overlap)), int(i * cropsize * (1 - overlap)),
                #                                  cropsize, cropsize)
                # out_band2 = in_band2.ReadAsArray(int(j * cropsize * (1 - overlap)), int(i * cropsize * (1 - overlap)),
                #                                  cropsize, cropsize)
                # out_band3 = in_band3.ReadAsArray(int(j * cropsize * (1 - overlap)), int(i * cropsize * (1 - overlap)),
                #                                  cropsize, cropsize)
                #
                # gtif_driver = gdal.GetDriverByName("GTiff")  # 数据类型，计算需要多大内存空间
                # filename = out_path + "/" + str(num) + '.png'  # 文件名称
                # out_ds = gtif_driver.Create(filename, cropsize, cropsize, 3, in_band1.DataType)
                # print("create new tif file succeed")
                #
                # top_left_x1 = top_left_x + int(j * cropsize * (1 - overlap)) * w_e_pixel_resolution
                # top_left_y1 = top_left_y + int(i * cropsize * (1 - overlap)) * n_s_pixel_resolution
                #
                # dst_transform = (top_left_x1, ori_transform[1], ori_transform[2], top_left_y1, ori_transform[4],
                #                  ori_transform[5])
                # out_ds.SetGeoTransform(dst_transform)
                #
                # # 设置SRS属性（投影信息）
                # out_ds.SetProjection(in_ds.GetProjection())
                #
                # # 写入目标文件（如果波段数有更改，这儿也需要修改）
                # out_ds.GetRasterBand(1).WriteArray(out_band1)
                # out_ds.GetRasterBand(2).WriteArray(out_band2)
                # out_ds.GetRasterBand(3).WriteArray(out_band3)
                #
                # # 将缓存写入磁盘，直接保存到了程序所在的文件夹
                # out_ds.FlushCache()
                # print(f"FlushCache succeed{num}")
                # del out_ds
                cropped.save('E:/20m/%d.png' % num)

        #  向前裁剪最后的列
        # for i in range(int(height / (cropsize_h * (1 - overlap)))):
        #     cropped = src_img.crop((width - cropsize_w, int(i * cropsize_h * (1 - overlap)),
        #                             width, int(i * cropsize_h * (1 - overlap) + cropsize_h)))
        #     num = num + 1
        #     arrSlope.append([width - cropsize_w, int(i * cropsize_h * (1 - overlap)), cropsize_w, cropsize_h])
        #     cropped.save('E:/crop_data_10m/%d.png' % num)

        # 向前裁剪最后的行
        # for j in range(int(width / (cropsize * (1 - overlap)))):
        #     cropped = src_img.crop((int(j * cropsize_w * (1 - overlap)), height - cropsize_h,
        #                             int(j * cropsize_w * (1 - overlap) + cropsize_w), height))
        #     num = num + 1
        #     arrSlope.append([int(j * cropsize_w * (1 - overlap)), height - cropsize_h, cropsize_w, cropsize_h])
        #     cropped.save('E:/crop_data_10m/%d.png' % num)

        np.array(arrSlope, dtype=int)
        # cropped.save('crop_data_15/%d.tif' % num)
        # np.savetxt("area_axis.txt", arrSlope)  # 保存文件
        return arrSlope, ranks_area, [width, height]

        # if mode == 'augment':
        #     src_roi, label_roi = data_augment(src_roi, label_roi)


if __name__ == '__main__':
    arr_Slope = creat_dataset()[0]
    # print(arr_Slope)
    # b = np.loadtxt("area_axis.txt", delimiter=',', dtype=float)  # 读取文件
    # print(b)
