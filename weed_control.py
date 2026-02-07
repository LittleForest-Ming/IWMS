import cv2
import numpy as np
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class AdvancedWeedAnalyzer:
    def __init__(self, img_path: str, json_path: str):
        self.img_path = img_path
        self.json_path = json_path

        # 1. 读取图像
        self.original_image = cv2.imread(img_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图片，请检查路径: {img_path}")

        self.height, self.width = self.original_image.shape[:2]
        self.bboxes = self._load_rois()

    def _load_rois(self) -> List[Dict]:
        """ 解析JSON获取目标框 (兼容性处理) """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            boxes = []
            idx = 0
            # 适配 LabelMe
            if 'shapes' in data:
                for shape in data['shapes']:
                    points = np.array(shape['points'])
                    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
                    boxes.append({'id': idx, 'rect': (x, y, w, h)})
                    idx += 1
            # 适配通用列表
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, list) and len(item) == 4:
                        boxes.append({'id': idx, 'rect': tuple(item)})
                    elif isinstance(item, dict) and 'bbox' in item:
                        boxes.append({'id': idx, 'rect': tuple(item['bbox'])})
                    idx += 1
            return boxes
        except Exception as e:
            print(f"[Warning] JSON读取失败或格式不匹配: {e}")
            return []

    def _preprocess_lighting(self, image):
        """
        【新增】光照预处理：CLAHE (对比度受限自适应直方图均衡化)
        解决阴影下杂草无法识别的问题
        """
        # 转换到 LAB 空间，只对 L (亮度) 通道做均衡化，不改变颜色信息
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def extract_green_lab_method(self) -> np.ndarray:
        """
        【重大改进】基于 Lab 空间的绿色提取
        相比 HSV，Lab 的 'a' 通道更能分离绿色与土壤，且受光照影响小。
        """
        # 1. 先做光照增强
        enhanced_img = self._preprocess_lighting(self.original_image)

        # 2. 转 Lab 空间
        lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)

        # 3. 提取 'a' 通道 (绿色-红色 分量)
        # 在 OpenCV Lab 中，a 通道范围 0-255。
        # 绿色通常在 a < 128 的区域 (数值越小越绿)，红色/土壤在 a > 128。
        _, a, _ = cv2.split(lab)

        # 4. 阈值分割
        # 这里的 115 是经验值，小于 115 的通常是绿色植物。
        # 你可以微调这个值：越小越严格（只选深绿），越大越宽松（包含黄绿）。
        _, binary = cv2.threshold(a, 115, 255, cv2.THRESH_BINARY_INV)

        # 5. 限制在标注框 ROI 区域内
        roi_mask_global = np.zeros((self.height, self.width), dtype=np.uint8)
        for box in self.bboxes:
            x, y, w, h = box['rect']
            cv2.rectangle(roi_mask_global, (x, y), (x + w, y + h), 255, -1)

        valid_binary = cv2.bitwise_and(binary, binary, mask=roi_mask_global)

        # 6. 形态学去噪 (去除雪花点)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 开运算去除噪点
        clean_mask = cv2.morphologyEx(valid_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # 闭运算填充叶片空洞
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return clean_mask

    def analyze_spatial_strategies(self, green_mask, overlap_tolerance=0.2):
        """ 保持逻辑不变：图论分析粘连关系 """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        G = nx.Graph()
        for box in self.bboxes:
            G.add_node(box['id'])

        box_strategies = {box['id']: 'adaptive' for box in self.bboxes}
        merged_regions = []

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50: continue  # 忽略过小噪点

            blob_mask = (labels == i).astype(np.uint8) * 255
            touched = []

            for box in self.bboxes:
                bx, by, bw, bh = box['rect']
                # 粗略判断是否相交
                if (stats[i, cv2.CC_STAT_LEFT] >= bx + bw or
                        stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] <= bx or
                        stats[i, cv2.CC_STAT_TOP] >= by + bh or
                        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] <= by):
                    continue

                # 精确计算交集
                box_mask = np.zeros_like(blob_mask)
                cv2.rectangle(box_mask, (bx, by), (bx + bw, by + bh), 255, -1)
                inter = cv2.bitwise_and(blob_mask, box_mask)
                if cv2.countNonZero(inter) > 0:
                    touched.append(box['id'])

            if len(touched) > 1:
                # 简单的完全连接图
                for u in touched:
                    for v in touched:
                        if u != v: G.add_edge(u, v)

        visited = set()
        for component in nx.connected_components(G):
            c_ids = list(component)
            if len(c_ids) == 1:
                box_strategies[c_ids[0]] = 'adaptive'
            else:
                # 聚合模式
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = 0, 0
                for bid in c_ids:
                    box_strategies[bid] = 'merged_dense'
                    bx, by, bw, bh = [b['rect'] for b in self.bboxes if b['id'] == bid][0]
                    min_x, min_y = min(min_x, bx), min(min_y, by)
                    max_x, max_y = max(max_x, bx + bw), max(max_y, by + bh)
                merged_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))

        return box_strategies, merged_regions

    def generate_smooth_density_map(self, green_mask, box_strategies, merged_regions):
        """
        【改进】动态计算网格大小 + 结果平滑处理
        """
        density_map = np.zeros((self.height, self.width), dtype=np.float32)

        # 1. 动态设定精细网格大小 (例如：取图片宽度的 1/40)
        # 这样 4000px 的图，网格约 100px，不会太细碎
        dynamic_grid_size = max(30, int(self.width / 40))
        print(f"[Info] 动态计算的网格大小: {dynamic_grid_size} px")

        # 策略A: 单体适应 (整个框一个密度)
        for box in self.bboxes:
            if box_strategies[box['id']] == 'adaptive':
                x, y, w, h = box['rect']
                roi = green_mask[y:y + h, x:x + w]
                if w * h > 0:
                    density = cv2.countNonZero(roi) / (w * h)
                    density_map[y:y + h, x:x + w] = density

        # 策略B: 密集精细网格
        for (mx, my, mw, mh) in merged_regions:
            step = dynamic_grid_size
            y_end = min(my + mh, self.height)
            x_end = min(mx + mw, self.width)

            for y in range(my, y_end, step):
                for x in range(mx, x_end, step):
                    cw = min(step, x_end - x)
                    ch = min(step, y_end - y)
                    roi = green_mask[y:y + ch, x:x + cw]
                    if cw * ch > 0:
                        local_density = cv2.countNonZero(roi) / (cw * ch)
                        density_map[y:y + ch, x:x + cw] = local_density

        return density_map

    def visualize(self, green_mask, density_map):
        """
        【改进】视觉效果优化
        """
        # 1. 对密度图进行高斯模糊，让它看起来像平滑的热力图，而不是马赛克
        # 根据网格大小决定模糊核大小
        blur_ksize = int(self.width / 80) | 1  # 确保是奇数
        smooth_density = cv2.GaussianBlur(density_map, (blur_ksize, blur_ksize), 0)

        # 归一化并上色
        heatmap_vis = (smooth_density * 255).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        # 混合显示
        mask_vis = density_map > 0.01
        overlay = self.original_image.copy()
        overlay[mask_vis] = heatmap_vis[mask_vis]
        final_blend = cv2.addWeighted(self.original_image, 0.6, overlay, 0.4, 0)

        # 绘图
        plt.figure(figsize=(16, 6))

        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(132)
        plt.title("Improved Green Mask (Lab Space)")
        plt.imshow(green_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(133)
        plt.title("Smoothed Density Heatmap")
        plt.imshow(cv2.cvtColor(final_blend, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# ================= 运行部分 =================
if __name__ == "__main__":
    # 自动获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 假设图片在 image 文件夹下
    img_name = "1-4"
    img_path = os.path.join(current_dir, "image", f"{img_name}.png")
    json_path = os.path.join(current_dir, "image", f"{img_name}.json")

    print(f"正在读取: {img_path}")

    if os.path.exists(img_path):
        try:
            analyzer = AdvancedWeedAnalyzer(img_path, json_path)

            # 1. 提取
            print("正在提取绿色...")
            green_mask = analyzer.extract_green_lab_method()

            # 2. 分析
            print("正在分析空间结构...")
            box_strategies, merged_regions = analyzer.analyze_spatial_strategies(green_mask)

            # 3. 生成
            print("正在生成热图...")
            density_map = analyzer.generate_smooth_density_map(green_mask, box_strategies, merged_regions)

            # 4. 显示
            analyzer.visualize(green_mask, density_map)

        except Exception as e:
            import traceback

            traceback.print_exc()
    else:
        print(f"❌ 找不到文件: {img_path}")