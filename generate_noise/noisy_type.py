import cv2
import numpy as np
import random
import os
import torch
from torchvision import transforms

random.seed(314)

"""
代码下这两个分别用来膨胀/腐蚀mask, src_label_np是原始mask, radius是膨胀/腐蚀kernel的半径
dilate_mask(src_label_np, radius)
erode_mask(src_label_np, radius)

不过这两个接口对应的是单个类别的, 如果是多类, 则需要把多类mask变成多个单类mask, 经过这个接口, 再合成多类mask

需要统计的是 
1 以pixel为单位,百分之多少的pixel对应的是错的标签.
2 以frame为单位,百分之多少的frame标签被改过了.
"""
# 扩张
def dilate_mask(src_label_np, radius): 
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param radius:
    :return:
    """
    dilation_diameter = int(2 * radius + 1)
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [radius, radius])) <= radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.dilate(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np

# 侵蚀
def erode_mask(src_label_np, radius):
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param radius:
    :return:
    """
    erode_diameter = int(2 * radius + 1)
    kernel = np.zeros((erode_diameter, erode_diameter), np.uint8)

    for row_idx in range(erode_diameter):
        for column_idx in range(erode_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [radius, radius])) <= radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.erode(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np


def foreground_move(src_label_np, max_distance):
    assert max_distance >= 0

    height, width = src_label_np.shape

    padded_label_np = np.pad(src_label_np, ((max_distance, max_distance), (max_distance, max_distance)), 'constant', constant_values=0)

    r = np.random.randint(0, 2 * max_distance)
    c = np.random.randint(0, 2 * max_distance)

    dst_label_np = padded_label_np[r:r + height, c:c + width]

    assert src_label_np.shape == dst_label_np.shape

    return dst_label_np


import skimage
from skimage import util


def fill_hole(im_in):
    im_floodfill = im_in.copy()

    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


def single_object_polygon_label_generation(label, num_points=8, shifting_ratio=0.2):
    # label has to be a 2D image
    assert len(label.shape) == 2
    label[label > 0] = 255
    label = label.astype(np.uint8)

    polygon_label = np.zeros_like(label)

    # fill holes
    label = fill_hole(label)

    # extracting the points of its contour
    contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    contour = contours[0]  # [x, y]
    num_contour_points = contour.shape[0]

    # generating target contour with num_points points
    target_contour_point_indexes = np.linspace(0, num_contour_points - 1, num_points)
    target_contour_point_indexes = [int(i) for i in target_contour_point_indexes]
    target_contour_points = contour[target_contour_point_indexes]

    # shifting target points
    if shifting_ratio > 0:
        height, width = label.shape[:]
        xy_min = target_contour_points.squeeze().min(axis=0)
        xy_max = target_contour_points.squeeze().max(axis=0)
        annotation_width = xy_max[0] - xy_min[0]
        annotation_height = xy_max[1] - xy_min[1]
        ratio = 0.2
        valid_x_range = int(ratio * annotation_width)
        valid_y_range = int(ratio * annotation_height)

        for idx in range(target_contour_points.shape[0]):
            x_shifting = random.randint(-valid_x_range, valid_x_range)
            y_shifting = random.randint(-valid_y_range, valid_y_range)
            target_contour_points[idx, 0, 0] += x_shifting
            target_contour_points[idx, 0, 1] += y_shifting
            target_contour_points[:, :, 0] = np.clip(target_contour_points[:, :, 0], 0, width - 1)
            target_contour_points[:, :, 1] = np.clip(target_contour_points[:, :, 1], 0, height - 1)

    # constructing polygon mask
    target_points_np = np.array(target_contour_points, np.int32)
    polygon_label = cv2.fillPoly(polygon_label, [target_points_np], 255)

    return polygon_label


def multi_object_polygon_label_generation(label, num_points=8, shifting_ratio=0.2):
    # This image has to contain less than 2 unique values
    unique_values = np.unique(label)
    num_unique_values = len(unique_values)
    assert num_unique_values <= 2
    if num_unique_values == 1:
        return

    label_value = unique_values[1]
    polygon_label = np.zeros_like(label)

    num_instances, labeled_result, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=8)

    # iterating each instance
    for instance_idx in range(1, num_instances):
        label_single = np.zeros_like(label)
        label_single[labeled_result == instance_idx] = 1
        polygon_label_single = single_object_polygon_label_generation(label_single, num_points, shifting_ratio)
        polygon_label[polygon_label_single > 0] = label_value

    return polygon_label


def multi_affine_noise_generation(label, max_rotation_degree=10, max_translate=0.1, max_rescale=0.1):
    # This image has to contain less than 2 unique values
    unique_values = np.unique(label)
    num_unique_values = len(unique_values)
    assert num_unique_values <= 2
    if num_unique_values == 1:
        return

    label_value = unique_values[1]
    polygon_label = np.zeros_like(label)

    num_instances, labeled_result, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=8)

    # iterating each instance
    for instance_idx in range(1, num_instances):
        label_single = np.zeros_like(label)
        label_single[labeled_result == instance_idx] = 1
        polygon_label_single = single_affine_noise_generation(label_single, max_rotation_degree, max_translate, max_rescale)
        polygon_label[polygon_label_single > 0] = label_value

    return polygon_label


def single_affine_noise_generation(label, max_rotation_degree=10, max_translate=0.1, max_rescale=0.1):
    assert len(label.shape) == 2

    label_tensor = torch.FloatTensor(label).unsqueeze(dim=0)
    # https://blog.csdn.net/lichaoqi1/article/details/123889172
    label_tensor = transforms.RandomAffine(degrees=(-max_rotation_degree, max_rotation_degree),
                                           translate=(0, max_translate),
                                           scale=(1 - max_rescale, 1 + max_rescale))(label_tensor)

    label = label_tensor.numpy().squeeze()

    return label


def add_noise(src_label_np, min_radius, max_radius, min_num_rays, max_num_rays, max_rotation_degree, max_translate, max_rescale):
    assert len(src_label_np.shape) == 2
    assert 0 <= max_radius <= max_radius

    random_num = random.random()

    radius = random.randint(min_radius, max_radius) #############################
    num_rays = random.randint(min_num_rays, max_num_rays)

    if random_num < 0.25:
        dst_label_np = dilate_mask(src_label_np, radius)
        noise_type = 'dilate'
    elif random_num < 0.50:
        dst_label_np = erode_mask(src_label_np, radius)
        noise_type = 'erode'
    elif random_num < 0.75:
        dst_label_np = multi_affine_noise_generation(src_label_np, max_rotation_degree, max_translate, max_rescale)
        noise_type = 'affine'
    else: 
        dst_label_np = multi_object_polygon_label_generation(src_label_np, num_rays)
        noise_type = 'polygon'

    return dst_label_np, noise_type

def add_noise_batch(src_label_np, min_radius, max_radius, min_num_rays, max_num_rays, max_rotation_degree, max_translate, max_rescale):
    # assert len(src_label_np.shape) == 2
    assert 0 <= max_radius <= max_radius

    random_num = random.random()

    radius = random.randint(min_radius, max_radius) #############################
    num_rays = random.randint(min_num_rays, max_num_rays)

    dst_label_np = []

    if random_num < 0.25:
        for i in src_label_np:
            label = dilate_mask(i, radius)
            dst_label_np.append(label)

        noise_type = 'dilate'
    elif random_num < 0.50:
        for i in src_label_np:
            label = erode_mask(i, radius)
            dst_label_np.append(label)

        noise_type = 'erode'
    elif random_num < 0.75:
        for i in src_label_np:
            label = multi_affine_noise_generation(i, max_rotation_degree, max_translate, max_rescale)
            dst_label_np.append(label)

        noise_type = 'affine'
    else: 
        for i in src_label_np:
            label = multi_object_polygon_label_generation(i, num_rays)
            dst_label_np.append(label)

        noise_type = 'polygon'

    return dst_label_np, noise_type

def add_noise_wo_polygon(src_label_np, min_radius, max_radius, max_rotation_degree, max_translate, max_rescale):
    assert len(src_label_np.shape) == 2
    assert 0 <= max_radius <= max_radius

    random_num = random.random()

    radius = random.randint(min_radius, max_radius) #############################

    if random_num < 0.25:
        dst_label_np = dilate_mask(src_label_np, radius)
        noise_type = 'dilate'
    elif random_num < 0.50:
        dst_label_np = erode_mask(src_label_np, radius)
        noise_type = 'erode'
    else:
        dst_label_np = multi_affine_noise_generation(src_label_np, max_rotation_degree, max_translate, max_rescale)
        noise_type = 'affine'

    return dst_label_np, noise_type

def add_noise_wo_polygon_batch(src_label_np, min_radius, max_radius, max_rotation_degree, max_translate, max_rescale):
    # assert len(src_label_np.shape) == 2
    assert 0 <= max_radius <= max_radius

    random_num = random.random()

    radius = random.randint(min_radius, max_radius) #############################
    dst_label_np = []

    if random_num < 0.25:
        for i in src_label_np:
            label = dilate_mask(i, radius)
            dst_label_np.append(label)

        noise_type = 'dilate'
    elif random_num < 0.50:
        for i in src_label_np:
            label = erode_mask(i, radius)
            dst_label_np.append(label)

        noise_type = 'erode'
    else:
        for i in src_label_np:
            label = multi_affine_noise_generation(i, max_rotation_degree, max_translate, max_rescale)
            dst_label_np.append(label)
            
        noise_type = 'affine'

    return dst_label_np, noise_type

if __name__ == '__main__':
    ################ Test the polygon_label
    mask_path = './MSS-TFAL/Endovis18/train/noisy_scene_labels_sep_mask_v0/seq_7/frame065/frame065_3_no_noise.png'
    src_label_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    polygon_label = multi_object_polygon_label_generation(src_label_np)
    print('polygon_label', polygon_label.shape)
    print('unique value in polygon label', np.unique(polygon_label))
    cv2.imwrite(os.path.join('polygon.png'), polygon_label)


