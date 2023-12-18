#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import open3d as o3d
import numpy as np


def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            return data
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None


def txt_to_dict(file_path):
    result_dict = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(':')
                result_dict[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"File not found at {file_path}")

    return result_dict


# mask 값이 True인 dp만 추출
def extract_data_by_mask(mask, data):
    # print("oooooo")
    # print("mask : ", mask)
    # print("data : ", data)
    filtered_data = {key: value for key, value in data.items() if mask[key] == 'True'}
    # print('filtered mask :', filtered_data)
    return filtered_data


# 데이터 중에 value 리스트의 값이 전부 0인 애들
def find_keys_with_zero_values(data_dict):
    keys_with_zero_values = []

    for key, value in data_dict.items():
        value_list = [float(item.strip('[]')) for item in value.split(',')]
        all_zero = all(float(item) == 0.0 for item in value_list)
        if all_zero:
            keys_with_zero_values.append(key)

    return keys_with_zero_values


# 데이터 중에 value 리스트의 값이 하나라도 0이 아닌 애들
def find_keys_with_non_zero_values(data_dict):
    keys_with_non_zero_values = []

    for key, value in data_dict.items():
        # 문자열을 리스트로 변환
        value_list = [float(item.strip('[]')) for item in value.split(',')]
        any_non_zero = any(item != 0.0 for item in value_list)
        if any_non_zero:
            keys_with_non_zero_values.append(key)

    return keys_with_non_zero_values


# dp 결과 value list에 저장된 grads를 다 더한 것.
def sum_values_in_dict(original_dict):
    result_dict = {key: sum(eval(value)) for key, value in original_dict.items()}
    return result_dict


# grads의 합이 작은 순서대로 정렬
def sort_dict_by_values(original_dict):
    sorted_dict = {k: v for k, v in sorted(original_dict.items(), key=lambda item: item[1])}
    return sorted_dict


# grad 크기 순서대로 정렬된 key값으로부터 origin xyz 좌표를 뽑는 함수
def get_dict_values_in_order(original_dict, keys):
    result_values = [original_dict[key] for key in keys if key in original_dict]
    return result_values


def save_grads_ply(position_th, th, path):
    th = str(th)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    color = [r, g, b]
    colors = [color for _ in range(len(position_th))]

    # 좌표를 PointCloud 객체로 변환
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(position_th)
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors) / 255.0)  # 0-255 RGB 값을 0-1 사이 값으로 변환

    # PLY 파일로 저장
    o3d.io.write_point_cloud("{}/th_{}_point_cloud.ply".format(path, th), point_cloud)


def save_origin_ply(position_th, path):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    color = [r, g, b]
    colors = [color for _ in range(len(position_th))]

    # 좌표를 PointCloud 객체로 변환
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(position_th)
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors) / 255.0)  # 0-255 RGB 값을 0-1 사이 값으로 변환

    # PLY 파일로 저장
    o3d.io.write_point_cloud("{}/ori_point_cloud.ply".format(path), point_cloud)


def txt2ply(path, th, ze):
    origin_xyz_path = path + '/origin_xyz_7005.txt'
    result_dict_path = path + '/result_dict_20000_7005.txt'
    mask_path = path + '/mask_20000_7005.txt'

    origin_xyz = txt_to_dict(origin_xyz_path)
    mask = txt_to_dict(mask_path)
    origin_xyz_position = list(origin_xyz.values())
    # print("LENGTH OF POSITIONS : ", len(origin_xyz_position))
    dp = txt_to_dict(result_dict_path)
    dp_mask = extract_data_by_mask(mask, dp)
    sum_dp_mask = sum_values_in_dict(dp_mask)

    # {'index' : grads 형태
    sorted_dp_mask = sort_dict_by_values(sum_dp_mask)

    # th = 0.3이면 하위 30%
    values = list(sorted_dp_mask.values())
    lower_30_percent_index = int(th * len(values))
    lower_30_percent_values = values[:lower_30_percent_index]
    th = lower_30_percent_values[-1]
    # Grads th 밑으로 변화없는 point 개수
    if ze:
        count_th_values = sum(1 for value in sorted_dp_mask.values() if value == 0)
        th = 0
    else:
        count_th_values = sum(1 for value in sorted_dp_mask.values() if value < th)
    # print("********")
    # print("Threshold count : ", count_th_values)

    sort_keys_mask = list(sorted_dp_mask.keys())
    sort_keys_mask_th = sort_keys_mask[:count_th_values]
    xyzs_th = get_dict_values_in_order(origin_xyz, sort_keys_mask_th)
    position_th = [list(map(float, coord.strip('[]').split(','))) for coord in xyzs_th]
    ori_position = [list(map(float, coord.strip('[]').split(','))) for coord in origin_xyz_position]
    save_grads_ply(position_th, th, path)
    save_origin_ply(ori_position, path)


def make_th_mask(path, xyz_len, random_v, iteration):
    th = 0.3
    iteration = iteration - 1

    origin_xyz_path = path + '/origin_xyz_{}.txt'.format(iteration)
    result_dict_path = path + '/result_dict_{}_{}.txt'.format(random_v, iteration)
    mask_path = path + '/mask_{}_{}.txt'.format(random_v, iteration)
    # print('^^^^^^^^^')
    # print('^^^^^^^^^')

    origin_xyz = txt_to_dict(origin_xyz_path)
    mask = txt_to_dict(mask_path)
    origin_xyz_position = list(origin_xyz.values())
    dp = txt_to_dict(result_dict_path)


    ## error 발생 부분
    dp_mask = extract_data_by_mask(mask, dp)
    sum_dp_mask = sum_values_in_dict(dp_mask)

    # {'index' : grads 형태
    sorted_dp_mask = sort_dict_by_values(sum_dp_mask)

    values = list(sorted_dp_mask.values())
    lower_30_percent_index = int(th * len(values))
    lower_30_percent_values = values[lower_30_percent_index-1]

    sort_keys_mask = list(sorted_dp_mask.keys())
    sort_keys_mask_th = sort_keys_mask[:lower_30_percent_index]
    # print(sort_keys_mask_th[:30])
    my_list = [False for _ in range(xyz_len)]
    # print("&&&&&&&")
    # print("my list len : ", len(my_list))
    for i in sort_keys_mask_th:
        my_list[int(i)] = True
    tensor_from_list = torch.tensor(my_list)
    return tensor_from_list


def make_zero_mask(path, xyz_len, random_v, iteration):
    iteration = iteration - 1
    origin_xyz_path = path + '/origin_xyz_{}.txt'.format(iteration)
    result_dict_path = path + '/result_dict_{}_{}.txt'.format(random_v, iteration)
    mask_path = path + '/mask_{}_{}.txt'.format(random_v, iteration)

    origin_xyz = txt_to_dict(origin_xyz_path)
    mask = txt_to_dict(mask_path)
    origin_xyz_position = list(origin_xyz.values())

    dp = txt_to_dict(result_dict_path)
    dp_mask = extract_data_by_mask(mask, dp)
    sum_dp_mask = sum_values_in_dict(dp_mask)

    # {'index' : grads 형태
    sorted_dp_mask = sort_dict_by_values(sum_dp_mask)
    count_th_values = sum(1 for value in sorted_dp_mask.values() if value == 0)

    sort_keys_mask = list(sorted_dp_mask.keys())
    sort_keys_mask_zero = sort_keys_mask[:count_th_values]
    my_list = [False for _ in range(xyz_len)]
    for i in sort_keys_mask_zero:
        my_list[int(i)] = True
    tensor_from_list = torch.tensor(my_list)
    return tensor_from_list


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)

    parser.add_argument("--th", type=float, default=0.3)
    parser.add_argument("--zero", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    print("CONVERTING... " + args.model_path)

    path = args.model_path
    threshold = args.th
    zero = args.zero
    print('Threshold : ',threshold)

    # Start GUI server, configure and run training
    txt2ply(path, threshold, zero)
    # All done
    print("\nCONVERT complete.")
