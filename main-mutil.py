#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         main
# Description:
# Author:       Ming_King
# Date:         2022/7/13
# -------------------------------------------------------------------------------
# 导入需要的库
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import tool_EXG
import crop
from osgeo import gdal

# 初始化目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_img_size,
                           non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device, time_sync

# 导入letterbox
from utils.augmentations import letterbox

weights = ROOT / 'weights/v5s.pt'  # 权重文件地址   .pt文件
source = ROOT / 'data/images'  # 测试数据文件(图片或视频)的保存路径
data = ROOT / 'data/coco128.yaml'  # 标签文件地址   .yaml文件

imgsz = (640, 640)  # 输入图片的大小 默认640(pixels)
conf_thres = 0.25  # object置信度阈值 默认0.25  用在nms中
iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
max_det = 1000  # 每张图片最多的目标数量  用在nms中
device = '0'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
dnn = False  # 使用OpenCV DNN进行ONNX推理

# 获取设备
device = select_device(device)

# 载入模型
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # 检查图片尺寸

# Half
# 使用半精度 Float16 推理
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()


def detect(img):
    # Dataloader
    # 载入数据
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # Run inference
    # 开始预测
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # 对图片进行处理
    im0 = img
    # Padded resize
    im = letterbox(im0, imgsz, stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    # Inference
    # 预测
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    # 用于存放结果
    detections_arr = []
    # Process predictions
    for i, det in enumerate(pred):  # per image 每张图片
        seen += 1
        # im0 = im0s.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            # 写入结果
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]]  # 检测到目标位置，格式：（left，top，w，h）
                # cls = names[int(cls)]
                xywh_id = [xywh[0], xywh[1], xywh[2], xywh[3], int(cls)]
                conf = float(conf)
                if conf < 0.10:
                    continue
                detections_arr.append(xywh_id)
    # 输出结果
    for i in detections_arr:
        print(i)
    # 推测的时间
    LOGGER.info(f'({t3 - t2:.3f}s)')
    return detections_arr


#  按特定比例缩放框架 (此方法适合图片,mp4视频和实时视频)
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)  # 宽度比例调整
    height = int(frame.shape[0] * scale)  # 高度比例调整
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Function to extract the file prefix
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


# 检测
def measure(dir_path, width, height):
    dirs = os.listdir(dir_path)  # 读取所有的文件
    # dirs = sorted(dirs)  # 文件字符串排序（supply）
    # Sort the list of files by file prefix
    dirs = get_file_prefix(dirs)

    heatmap_mask = tool_EXG.create_img(int(width / 8), int(height / 8), if_rgb=False, dtype="int")
    # count = 0
    for file in dirs:
        if os.path.splitext(file)[1] == '.png':  # 只读取固定后缀的文件
            file_pname = os.path.splitext(file)[0]
            print(file_pname)
            file_path = str(dir_path+'/'+file)
            image = cv2.imread(file_path)
            target_ayy = detect(image)
            img, weed_midpoints, maize_midpoints, weed_pixs = tool_EXG.separate_2dif(image, target_ayy,
                                                                                     if_weedpixs=True)
            # 获取目标经纬度坐标
            # laANDlo = pixs_convert_latitudeANDlo(file_path, weed_midpoints, maize_midpoints)
            # current_roi_laANDlo = [int(file_pname), float(','.join(str(x) for x in laANDlo))]
            # all_laANDlo.append(pixs_convert_latitudeANDlo(file_path, weed_midpoints, maize_midpoints))
            # with open('E:/test/' + '%d.txt' % int(file_pname), 'w+') as f:
            #     for i in range(len(current_roi_laANDlo)):
            #         f.write(str(current_roi_laANDlo[i]) + '\n')
            # f.close()

            create_heatmap_value(heatmap_mask, arr_Slope[int(file_pname)-1], weed_pixs)
            # contourf_arr[ranks_area[int(file_pname)][0]][ranks_area[int(file_pname)][1]] = weed_pixs/1024*255
            mask_points = tool_EXG.midpoints_plot(image, maize_midpoints, weed_midpoints, if_seeding=True, if_weed=False)
            # cv2.imshow('points', mask_points)
            # cv2.imshow('test_img', img)
            # cv2.waitKey(0)
            mosaic_img = tool_EXG.image_mosaic(big_img, mask_points, arr_Slope[int(file_pname)-1])
            # cv2.destroyAllWindows()
            # if count == 100:
            #     break
            # count += 1

            # 测试
            #############################################################
            # mosaic_img_2 = tool_EXG.image_mosaic(big_img_2, img, arr_Slope[int(file_pname) - 1])
            mask_points_2 = tool_EXG.midpoints_plot(image, maize_midpoints, weed_midpoints, if_seeding=False, if_weed=True)
            mosaic_img_3 = tool_EXG.image_mosaic(big_img_3, mask_points_2, arr_Slope[int(file_pname) - 1])

            # # 颜色区域映射矫正
            # color_img_amend = np.array(heatmap_mask / (np.max(heatmap_mask) - np.min(heatmap_mask)) * 255, dtype=np.uint8)
            # # 处方图生成
            # Color_image = cv2.applyColorMap(color_img_amend, cv2.COLORMAP_JET)
            # image_1 = rescaleFrame(mosaic_img_2, scale=0.1)
            # image_3 = rescaleFrame(Color_image, scale=0.1)
            # image_4 = rescaleFrame(mosaic_img, scale=0.1)
            # image_2 = rescaleFrame(mosaic_img_3, scale=0.1)
            # # 显示数据
            # imgStack = tool.stackImages(0.9, ([image_1, image_3], [image_4, image_2]))
            # cv2.imshow('Project1', imgStack)
            # cv2.waitKey(0)
            #############################################################
    return mosaic_img, mosaic_img_3, heatmap_mask
    # return mosaic_img, mosaic_img_2, mosaic_img_3, heatmap_mask
    # return heatmap_mask
    # return mosaic_img
    # return mosaic_img


# 获取目标经纬度坐标
def pixs_convert_latitudeANDlo(file_path, weed_midpoints, maize_midpoints, weed_and_maize=False):  # 默认得到点的经纬度坐标
    dataset = gdal.Open(file_path)  # 打开tif

    geo_information = dataset.GetGeoTransform()
    band = dataset.RasterCount
    dem = dataset.GetRasterBand(1).ReadAsArray()
    # 获取行列数，对应其经纬度,j对于x坐标
    ob_laANDlo = []
    if weed_and_maize:
        all_midpoints = weed_midpoints + maize_midpoints
        for (x, y) in all_midpoints:  # 获取所以目标经纬度坐标
            x, y = int(x), int(y)
            # 有效高程
            if dem[y][x] > 0:
                # 输出经纬度
                lon = geo_information[0] + x * geo_information[1] + y * geo_information[2]
                lat = geo_information[3] + x * geo_information[4] + y * geo_information[5]
                child = [lon, lat, dem[y][x], y, x]
                ob_laANDlo.append(child)
    else:
        for (x, y) in weed_midpoints:
            x, y = int(x), int(y)
            # 有效高程
            if dem[y][x] > 0:
                # 输出经纬度
                lon = geo_information[0] + x * geo_information[1] + y * geo_information[2]
                lat = geo_information[3] + x * geo_information[4] + y * geo_information[5]
                child = [lon, lat, dem[y][x], y, x]
                ob_laANDlo.append(child)
    return ob_laANDlo


# 生成处方热力图数值(采用数量或者面积比)
def create_heatmap_value(heatmap_mask, area_site, weed_number_or_pixs):
    heatmap_mask[int(area_site[1]/8):int(area_site[1]/8)+int(area_site[3]/8),
                int(area_site[0]/8):int(area_site[0]/8)+int(area_site[2]/8)] = int((weed_number_or_pixs+0.3)*255)


if __name__ == '__main__':
    dir_path = 'E:/20m'
    arr_Slope, ranks_area, [width, height] = crop.creat_dataset()
    all_laANDlo = []
    # contourf_arr = np.zeros((ranks_area[0][0], ranks_area[0][1]), dtype=int)  # 等高线图-数值

    big_img = tool_EXG.create_img(width+200, height+200, if_rgb=True)  # W=28853*H=49314

    # 测试
    big_img_2 = tool_EXG.create_img(width+200, height+200, if_rgb=True)  # W=28853*H=49314
    big_img_3 = tool_EXG.create_img(width+200, height+200, if_rgb=True)  # W=28853*H=49314

    # mosaic_image, heatmap_mask = measure(dir_path)
    mosaic_image_1, mosaic_image_3, heatmap_img = measure(dir_path, width=width, height=height)
    # heatmap_img = measure(dir_path)   # max_min也可采用np.max()||np.min()
    # print(np.max(heatmap_img))
    # 颜色区域映射矫正
    # amend_color_img = np.array(heatmap_img / (np.max(heatmap_img) - np.min(heatmap_img)) * 255, dtype=np.uint8)
    amend_color_img = np.array(heatmap_img, dtype=np.uint8)
    # 处方图生成
    color_image = cv2.applyColorMap(amend_color_img, cv2.COLORMAP_JET)
    color_image_resize = rescaleFrame(color_image, scale=0.1)
    cv2.imwrite('E:/small_heatmap.jpg', color_image)
    # cv2.imshow('heatmap', color_image)
    #  绘制等高线图
    # tool.plt_contourf(contourf_arr.shape[1], contourf_arr.shape[0], contourf_arr)

    img_resized = rescaleFrame(mosaic_image_1, scale=0.1)
    # cv2.imwrite('points_weed1.jpg', img_resized)
    # cv2.imshow('ms', img_resized)
    # cv2.waitKey(0)

    # 生成整体大图
    # img_all = tool_EXG.stackImages(1.0, ([mosaic_image_2, color_image], [mosaic_image_1, mosaic_image_3]))
    # cv2.imwrite('big.jpg', img_all)

    # 测试部分
    # img_resized_2 = rescaleFrame(mosaic_image_2, scale=0.1)
    # img_resized_3 = rescaleFrame(mosaic_image_3, scale=0.1)
    # 显示数据
    # imgStack = tool_EXG.stackImages(0.9, ([img_resized_2, color_image_resize], [img_resized, img_resized_3]))
    cv2.imshow('Project1', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

