import os
import numpy as np
import glob
import csv
import cv2
from config import BATCH_SIZE, INPUT_SIZE, BACKGROUND_RATIO, TRAINING_DATA_PATH, IS_CASE_IMAGE_SUFFIXES, MIN_CROP_SIDE_RATIO, RANDOM_SCALE, GEOMETRY, MIN_TEXT_SIZE, PLAT_FORM
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

'''
图片预处理工具类
'''

def polygon_area(poly):
    '''
    计算四边形面积
    params:
        poly: 单个多边形四个点坐标数组
    returns:
        返回当前四边形面积（为负数）
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def load_annoataion(p):
    '''
    加载标注文件信息
    params:
        p: 文件图片标注文件路径
    returns:
        返回当前标注文件的坐标信息和label真值信息（为true则是###，false为需要识别）
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r', encoding='UTF8') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def get_images(training_data_path):
    '''
    获取所有训练图片路径
    params:
        training_data_path: 训练图片路径
    returns: 
        返回文件路径数组集合
    '''
    files = []
    suffixes = ['jpg', 'png', 'jpeg']
    if not IS_CASE_IMAGE_SUFFIXES:
        suffixes.append('JPG')
    for ext in suffixes:
        files.extend(glob.glob(os.path.join('{}/images/'.format(training_data_path), '*.{}'.format(ext))))
    return files


def check_and_validate_polys(polys, tags, original_size):
    '''
    检查文本多边形是否是同一个方向并过滤不正确的四边形
    params:
        polys: 每一个标注文件中所有点的信息，shape: (num_poly, 4, 2)
        tags: 每一个标注文件中每一个四边形标签，True为不care，False为care
        original_size: 原始图片的高宽
    returns: 
        返回过滤后的多边形
    '''
    (h, w) = original_size
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)

def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    裁剪图片区域以筛选多边形，筛选满足1: 宽高大于设定ratio; 2: polys中满足在裁剪区域中
    params: 从图片中随机裁剪一个区域
        im: 缩放之后的图片数据
        polys: 经过缩放之后的多边形坐标信息
        tags: 图片标注True和Flase数组
        crop_background: 如果文字区域不在裁剪区域中，是否采取裁剪背景区域方式
        max_tries: 尝试找出裁剪区域最大尝试次数
    returns: 
        返回满足条件的多边形信息: 裁剪区域之后的图片数据, 满足条件的多边形数据, 对应多边形的tag
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    # 将所有多边形所在的维度信息在h_array, w_array置1
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32) # 四舍五入操作
        minx = np.min(poly[:, 0]) # 
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w : maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h : maxy+pad_h] = 1
    # 确保裁剪的区域不跨文本，不会截断裁剪文本区域
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < MIN_CROP_SIDE_RATIO * w or ymax - ymin < MIN_CROP_SIDE_RATIO * h:
            # 裁剪区域太小
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # 不在裁剪区中
            if crop_background:
                return im[ymin : ymax+1, xmin : xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue

        # im_source = im

        im = im[ymin : ymax+1, xmin : xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        
        # 图片显示裁剪区域
        # im_source = cv2.rectangle(im_source, (xmin, ymin), (xmax + 1, ymax + 1), (255, 0, 0), 2) # 画出裁剪区域

        # cv2.imshow('123', im_source)
        # cv2.waitKey(0)

        # 图片显示文本框坐标在裁剪区域，正好判断是否正确
        # for tmp in polys:
        #     im_tmp1 = cv2.rectangle(im, (tmp[0][0], tmp[0][1]), (tmp[2][0], tmp[2][1]), (0, 255, 0), 2) # 画出
        #     cv2.imshow('123', im_tmp1)
        #     cv2.waitKey(0)

        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # 计算从p3到p1-p2的距离
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def fit_line(p1, p2):
    # 拟合一条置先 ax + by + c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]

def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # 从跨点的线获取垂直线
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle



def rectangle_from_parallelogram(poly):
    '''
    拟合平行四边形的矩形
    prams:
        poly: 四边形坐标信息
    returns:

    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    '''
    排序多边形的四个坐标，多边形中的点应顺时针排序
    首先找到最低点
    '''
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbox(im_size, polys, tags, im=None):
    '''
    生成rbox数据
    当前图片已被进行裁剪
    params:
        im_size: 输入图片大小
        polys: 当前图片的多边形位置信息
        tags: 输入图片tag标记
        im: 图片数据（这边只是测试传输，该参数可以不需要）
    returns:

    '''
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # print('h = {}, w = {}'.format(str(h), str(w)))
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]

        # # 画出原图片上的矩形框（这边暂时用矩形框画）
        # im = cv2.rectangle(im, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), (0, 255, 255), 2)
        # # 画出缩小之后的四边形坐标，具体这边的逻辑可以参考论文那边的
        # im = cv2.rectangle(im, (shrinked_poly[0][0][0], shrinked_poly[0][0][1]), (shrinked_poly[0][2][0], shrinked_poly[0][2][1]), (0, 0, 255), 2)

        # cv2.imshow('line', im)
        # cv2.waitKey(0)



        cv2.fillPoly(score_map, shrinked_poly, 1) # 注意cv是按照高宽来处理的，所以shrinked_poly 是高宽
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < MIN_TEXT_SIZE:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # im = cv2.rectangle(im, (shrinked_poly[0][0][0], shrinked_poly[0][0][1]), (shrinked_poly[0][2][0], shrinked_poly[0][2][1]), (0, 0, 255), 2)
        # cv2.imshow('123', im)
        # cv2.waitKey(0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2 - parallel lines through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3 - after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask

def process_data(training_data_path=TRAINING_DATA_PATH, 
              input_size=INPUT_SIZE, 
              background_ratio=BACKGROUND_RATIO, 
              random_scale=np.array(RANDOM_SCALE), 
              vis=False):
    '''
    params:
        training_data_path: 训练图片路径
        input_size: 输入图片尺寸
        background_ratio: 
        random_scale: 图片缩放因子，需要缩放图片数据以及坐标数据
        vis: 是否显示
    returns:
        images: (None, 512, 512, 3), image_fns: (None,), score_maps: (None, 128, 128, 1), geo_maps: (None, 128, 128, 5), training_masks: (None, 128, 128, 1)
    '''
    image_list = np.array(get_images(training_data_path))
    print('{} training images in {}'.format(image_list.shape[0], training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, _ = im.shape
                # txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                if PLAT_FORM == 'WINDOWS':
                    txt_fn = os.path.join('{}/labels/gt_{}.txt'.format(training_data_path, im_fn.split('\\')[-1].split('.')[0]))
                elif PLAT_FORM == 'LINUX':
                    txt_fn = os.path.join('{}/labels/gt_{}.txt'.format(training_data_path, im_fn.split('/')[-1].split('.')[0]))
                else:
                    txt_fn = ''
                
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale

                # 从图片中随机裁剪一个区域
                # np.random.rand()
                if np.random.rand() < background_ratio:
                    # 裁剪背景区域
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        continue
                    # 填充并改变大小
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if GEOMETRY == 'RBOX' else 8
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    
                    # for poly in text_polys:
                    #     im = cv2.rectangle(im, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), (0, 255, 0), 2)

                    # cv2.imshow('123', im)
                    # cv2.waitKey(0)

                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = im.shape
                    # 将图像填充到训练输入尺寸或者图像较大的边
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # 将图像裁剪到后续输入到网络的尺寸
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape

                    # for poly in text_polys:
                    #     im = cv2.rectangle(im, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), (0, 255, 0), 2)

                    # cv2.imshow('123', im)
                    # cv2.waitKey(0)

                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags, im=im)

                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    axs[0, 0].imshow(im[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                # images.append(im[:, :, ::-1].astype(np.float32))
                images.append(im[:, :, ::-1])
                # cv2.imshow('123', im)
                # cv2.waitKey(0)
                
                # cv2.imshow('123', np.array(images[0]))
                # cv2.waitKey(0)
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32)) # 因为当前score_map
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                # if len(images) == batch_size:
                #     yield images, image_fns, score_maps, geo_maps, training_masks
                #     images = []
                #     image_fns = []
                #     score_maps = []
                #     geo_maps = []
                #     training_masks = []
                
                # images: (1, 512, 512, 3), image_fns: (1,), score_maps: (1, 128, 128, 1), geo_maps: (1, 128, 128, 5), training_masks: (1, 128, 128, 1)
                # return images, image_fns, score_maps, geo_maps, training_masks

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
    return images, image_fns, score_maps, geo_maps, training_masks


# def get_batch(training_data_path=TRAINING_DATA_PATH, 
#               input_size=INPUT_SIZE, 
#               batch_size=BATCH_SIZE, 
#               background_ratio=BACKGROUND_RATIO, 
#               random_scale=np.array(RANDOM_SCALE), 
#               vis=False):
#     '''
#     training_data_path: 训练图片路径
#     input_size: 输入图片尺寸
#     batch_size: 每一批次大小
#     background_ratio: 
#     random_scale: 图片缩放因子，需要缩放图片数据以及坐标数据
#     vis: 
#     '''
#     image_list = np.array(get_images(training_data_path))
#     print('{} training images in {}'.format(image_list.shape[0], training_data_path))
#     index = np.arange(0, image_list.shape[0])
#     while True:
#         np.random.shuffle(index)
#         images = []
#         image_fns = []
#         score_maps = []
#         geo_maps = []
#         training_masks = []
#         for i in index:
#             try:
#                 im_fn = image_list[i]
#                 im = cv2.imread(im_fn)
#                 h, w, _ = im.shape
#                 txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
#                 if not os.path.exists(txt_fn):
#                     print('text file {} does not exists'.format(txt_fn))
#                     continue
#                 text_polys, text_tags = load_annoataion(txt_fn)
#                 text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

#                 rd_scale = np.random.choice(random_scale)
#                 im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
#                 text_polys *= rd_scale

#                 # 从图片中随机裁剪一个区域
#                 # np.random.rand()
#                 if np.random.rand() < background_ratio:
#                     # 裁剪背景区域
#                     im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
#                     if text_polys.shape[0] > 0:
#                         continue
#                     # 填充并改变大小
#                     new_h, new_w, _ = im.shape
#                     max_h_w_i = np.max([new_h, new_w, input_size])
#                     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
#                     im_padded[:new_h, :new_w, :] = im.copy()
#                     im = cv2.resize(im_padded, dsize=(input_size, input_size))
#                     score_map = np.zeros((input_size, input_size), dtype=np.uint8)
#                     geo_map_channels = 5 if GEOMETRY == 'RBOX' else 8
#                     geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
#                     training_mask = np.ones((input_size, input_size), dtype=np.uint8)
#                 else:
#                     im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    
#                     # for poly in text_polys:
#                     #     im = cv2.rectangle(im, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), (0, 255, 0), 2)

#                     # cv2.imshow('123', im)
#                     # cv2.waitKey(0)

#                     if text_polys.shape[0] == 0:
#                         continue
#                     h, w, _ = im.shape
#                     # 将图像填充到训练输入尺寸或者图像较大的边
#                     new_h, new_w, _ = im.shape
#                     max_h_w_i = np.max([new_h, new_w, input_size])
#                     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
#                     im_padded[:new_h, :new_w, :] = im.copy()
#                     im = im_padded
#                     # 将图像裁剪到后续输入到网络的尺寸
#                     new_h, new_w, _ = im.shape
#                     resize_h = input_size
#                     resize_w = input_size
#                     im = cv2.resize(im, dsize=(resize_w, resize_h))
#                     resize_ratio_3_x = resize_w/float(new_w)
#                     resize_ratio_3_y = resize_h/float(new_h)
#                     text_polys[:, :, 0] *= resize_ratio_3_x
#                     text_polys[:, :, 1] *= resize_ratio_3_y
#                     new_h, new_w, _ = im.shape

#                     # for poly in text_polys:
#                     #     im = cv2.rectangle(im, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), (0, 255, 0), 2)

#                     # cv2.imshow('123', im)
#                     # cv2.waitKey(0)

#                     score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags, im=im)

#                 if vis:
#                     fig, axs = plt.subplots(3, 2, figsize=(20, 30))
#                     axs[0, 0].imshow(im[:, :, ::-1])
#                     axs[0, 0].set_xticks([])
#                     axs[0, 0].set_yticks([])
#                     for poly in text_polys:
#                         poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
#                         poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
#                         axs[0, 0].add_artist(Patches.Polygon(
#                             poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
#                         axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
#                     axs[0, 1].imshow(score_map[::, ::])
#                     axs[0, 1].set_xticks([])
#                     axs[0, 1].set_yticks([])
#                     axs[1, 0].imshow(geo_map[::, ::, 0])
#                     axs[1, 0].set_xticks([])
#                     axs[1, 0].set_yticks([])
#                     axs[1, 1].imshow(geo_map[::, ::, 1])
#                     axs[1, 1].set_xticks([])
#                     axs[1, 1].set_yticks([])
#                     axs[2, 0].imshow(geo_map[::, ::, 2])
#                     axs[2, 0].set_xticks([])
#                     axs[2, 0].set_yticks([])
#                     axs[2, 1].imshow(training_mask[::, ::])
#                     axs[2, 1].set_xticks([])
#                     axs[2, 1].set_yticks([])
#                     plt.tight_layout()
#                     plt.show()
#                     plt.close()

#                 # images.append(im[:, :, ::-1].astype(np.float32))
#                 images.append(im[:, :, ::-1])
#                 image_fns.append(im_fn)
#                 score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
#                 geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
#                 training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

#                 # if len(images) == batch_size:
#                 #     yield images, image_fns, score_maps, geo_maps, training_masks
#                 #     images = []
#                 #     image_fns = []
#                 #     score_maps = []
#                 #     geo_maps = []
#                 #     training_masks = []

#             except Exception as e:
#                 import traceback
#                 traceback.print_exc()
#                 continue

if __name__ == "__main__":
    images, image_fns, score_maps, geo_maps, training_masks = process_data()
    a = 10