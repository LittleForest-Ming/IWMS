# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# # -------------------------------------------------------------------------------
# # Name:         Draw_test
# # Description:
# # Author:       Ming_King
# # Date:         2023/3/27
# # -------------------------------------------------------------------------------
import cv2
import numpy as np


def iou(box1, box2):
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[3] - box2[1]) * (box2[4] - box2[2])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.4):
    matches = []
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if pred_box[0] == gt_box[0]:  # 检查类别是否匹配
                cur_iou = iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = idx

        if best_iou >= iou_threshold:
            matches.append((pred_box, gt_boxes[best_gt_idx], 'TP'))
            gt_boxes.pop(best_gt_idx)
        else:
            matches.append((pred_box, None, 'FP'))

    # 处理未匹配到的真实边界框（FN）
    for gt_box in gt_boxes:
        matches.append((None, gt_box, 'FN'))

    return matches


def create_color_mapping(class_names):
    base_colors = {
        'GT': (255, 0, 0),    # 蓝色
        'TP': (0, 255, 0),    # 绿色
        'FN': (0, 255, 255),  # 黄色
        'FP': (0, 0, 255)     # 红色
    }

    color_mapping = {}
    for idx, class_name in enumerate(class_names):
        class_colors = {}
        for key, color in base_colors.items():
            class_colors[key] = tuple(int(c * (1 - idx * 0.2)) for c in color)
        color_mapping[class_name] = class_colors

    return color_mapping


def draw_boxes(image, matches, color_mapping):
    for pred_box, gt_box, match_type in matches:
        class_name = pred_box[0] if pred_box is not None else gt_box[0]
        colors = color_mapping[str(class_name)]
        if pred_box is not None:
            x1, y1, x2, y2 = pred_box[1:]
            color = colors[match_type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if gt_box is not None:
            x1, y1, x2, y2 = gt_box[1:]
            color = colors[match_type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if gt_box is not None and match_type == 'TP':
            x1, y1, x2, y2 = gt_box[1:]
            color = colors['GT']
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image


def draw_detect_labels(image, pre_labels):
    colors = [(119, 172, 48), (180, 105, 255), (76, 175, 80), (0, 119, 255)]
    for cls, x1, y1, x2, y2 in pre_labels:
        color = colors[int(cls)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image


def main():
    # 假设这是您的原始图像
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    # 定义类别名称
    class_names = ['cat', 'dog']

    # 为每个类别创建颜色映射
    color_mapping = create_color_mapping(class_names)

    # 定义预测边界框和真实边界框
    pred_boxes = [
        ['cat', 50, 50, 150, 150],
        ['dog', 200, 200, 300, 300],
        ['cat', 400, 400, 500, 500],
        ['dog', 550, 550, 650, 650],
        ['cat', 15, 25, 35, 35],
        ['dog', 100, 220, 130, 230]
    ]

    gt_boxes = [
        ['cat', 55, 55, 155, 155],
        ['dog', 210, 210, 310, 310],
        ['cat', 410, 410, 510, 510],
        ['cat', 50, 200, 100, 250]
    ]

    # 匹配预测边界框和真实边界框
    matches = match_boxes(pred_boxes, gt_boxes)

    # 画出边界框
    draw_boxes(image, matches, color_mapping)

    # 显示图像
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()










# import numpy as np
#
# def nms(detections, iou_threshold):
#     if len(detections) == 0:
#         return []
#
#     classes = np.array([det[0] for det in detections])
#     boxes = np.array([det[1:] for det in detections], dtype=np.float32)
#
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#
#     unique_classes = np.unique(classes)
#     keep = []
#
#     for label in unique_classes:
#         label_indices = np.where(classes == label)[0]
#         label_boxes = boxes[label_indices]
#         label_areas = areas[label_indices]
#
#         order = label_areas.argsort()[::-1]
#
#         while order.size > 0:
#             i = order[0]
#             keep.append(label_indices[i])
#
#             xx1 = np.maximum(x1[label_indices[i]], x1[label_indices[order[1:]]])
#             yy1 = np.maximum(y1[label_indices[i]], y1[label_indices[order[1:]]])
#             xx2 = np.minimum(x2[label_indices[i]], x2[label_indices[order[1:]]])
#             yy2 = np.minimum(y2[label_indices[i]], y2[label_indices[order[1:]]])
#
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             intersection = w * h
#
#             iou = intersection / (label_areas[i] + label_areas[order[1:]] - intersection)
#
#             inds = np.where(iou <= iou_threshold)[0]
#             order = order[inds + 1]
#
#     return keep
# detections = [
#     (1, 50, 100, 200, 300),
#     (1, 55, 105, 210, 310),
#     (2, 600, 700, 800, 900),
#     (2, 610, 705, 815, 910)
# ]
#
# iou_threshold = 0.3
#
# # 使用NMS函数
# keep_indices = nms(detections, iou_threshold)
#
# # 输出结果
# print("Kept detections:")
# for index in keep_indices:
#     print(detections[index])
