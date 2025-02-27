#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         test
# Description:
# Author:       Ming_King
# Date:         2022/7/12
# -------------------------------------------------------------------------------
from osgeo import gdal

# ##############################读取图片信息###################################
# ##图片路径设置
in_ds = gdal.Open("D:/Data Set/result.tif")
out_path = "crop_data_15"

# 读取原图中的每个波段，通道数从1开始，默认前三波段
in_band1 = in_ds.GetRasterBand(1)
in_band2 = in_ds.GetRasterBand(2)
in_band3 = in_ds.GetRasterBand(3)

# 获取原图的原点坐标信息
ori_transform = in_ds.GetGeoTransform()
top_left_x = ori_transform[0]  # 左上角x坐标
top_left_y = ori_transform[3]  # 左上角y坐标
w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

# ##############################裁切信息设置###################################
# ##定义切图的起始点像素位置
offset_x = 0
offset_y = 0
# ##定义切图的大小（矩形框）
block_xsize = 1024  # 行
block_ysize = 1024  # 列
count = 0
# ##是否需要最后不足补充，进行反向裁切（裁切数量）
im_width = in_ds.RasterXSize  # 栅格矩阵的列数
im_height = in_ds.RasterYSize  # 栅格矩阵的行数
num_width = int(im_width/block_ysize)
num_height = int(im_height/block_xsize)
if True:
    if im_width % block_ysize == 0:  # 判断是否能完整裁切
        num_width = num_width
        wb = False
    else:
        num_width += 1
        wb = True
    if im_height % block_xsize == 0:
        num_height = num_height
        hb = False
    else:
        num_height += 1
        hb = True

# ##图像重叠区设置（暂时不考虑）

# ################################开始裁切#####################################

for i in range(num_width):
    offset_x1 = offset_x + block_xsize * i
    if hb and i == num_width - 1:
        offset_x1 = im_width - block_xsize
    for j in range(num_height):
        count = count+1

        offset_y1 = offset_y+block_ysize*j
        if wb and j == num_height - 1:
            offset_y1 = im_height - block_ysize
        out_band1 = in_band1.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
        out_band2 = in_band2.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
        out_band3 = in_band3.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)

        gtif_driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间，但是这儿是只有GTiff吗？
        filename = out_path + "/" + str(count)+'.png'  # 文件名称
        out_ds = gtif_driver.Create(filename, block_xsize, block_ysize, 3, in_band1.DataType)
        print("create new tif file succeed")

        top_left_x1 = top_left_x + offset_x1 * w_e_pixel_resolution
        top_left_y1 = top_left_y + offset_y1 * n_s_pixel_resolution

        dst_transform = (top_left_x1, ori_transform[1], ori_transform[2], top_left_y1, ori_transform[4], ori_transform[5])
        out_ds.SetGeoTransform(dst_transform)

        # 设置SRS属性（投影信息）
        out_ds.SetProjection(in_ds.GetProjection())

        # 写入目标文件（如果波段数有更改，这儿也需要修改）
        out_ds.GetRasterBand(1).WriteArray(out_band1)
        out_ds.GetRasterBand(2).WriteArray(out_band2)
        out_ds.GetRasterBand(3).WriteArray(out_band3)

        # 将缓存写入磁盘，直接保存到了程序所在的文件夹
        out_ds.FlushCache()
        print(f"FlushCache succeed{count}")
        del out_ds