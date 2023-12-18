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
import torch
from random import randint

from txt2ply import make_th_mask, make_zero_mask
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    global query_filter_data
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    ii = 0
    percent = 0.1
    random_v = 40000
    save_path = dataset.model_path


    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 7000이상부터 100회마다 0,1,2,3,4는 grads 계산, 5번째는
        # if iteration >= 7000 and (str(iteration)[-3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", '9'])\
        #         and (str(iteration)[-1] in ["1", "2", "3", "4", "5"]):


        if iteration >= 7000 and iteration % 100 in (1, 2, 3, 4, 5, 6, 7, 8):
            # print("!!!!!!!!!!!!!!!!!!!!!")
            # print("!!!!ITERATION {}!!!!!".format(iteration))
            # print("!!!!!!!!!!!!!!!!!!!!!")
            # print("====================")
            # print("====================")
            # print("====XYZ SIZE : {}===".format(gaussians.get_xyz.shape[0]))
            # print("====================")
            # print("====================")
            if ii == 0:
                gaussians.save_original_params()

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]

            # TODO : filtering visibility point
            # visibility filter : 255725개를 가진 [T, T, F ,T, ... ] 1차원 텐서
            # 여기서 True를 찍어보니   204734개가 나옴. False는 50991 개 -> True 중에 뽑아야겠다.
            # indexed_points = create_indexed_dict(gaussians.get_xyz) # -> GPU에서 할 방법을...
            visible_points = gaussians.get_xyz[visibility_filter]
            origin_xyz = gaussians.get_xyz
            origin_xyz_dict = tensor_to_dict(origin_xyz)
            if ii == 0:
                query_filter_data = random_query_filter(gaussians.get_xyz, random_v)


            # TODO : point move
            new_xyz = gaussians.move_specific_points(query_filter_data)
            dict_mask = gaussians.bool_mask_to_dict(query_filter_data)

            new_xyz_dict = tensor_to_dict(new_xyz)

            # TODO : render image
            image = render_pkg["render"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, (pipe, background))

                # print('Train')

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning

                    # TODO : gradient calculate using moved points
                    # Gradient 계산
                    gaussians.max_radii2D[query_filter_data] = torch.max(gaussians.max_radii2D[query_filter_data],
                                                                         radii[query_filter_data])
                    gaussians.add_densification_stats(viewspace_point_tensor, query_filter_data)

                    # TODO : archive gradient with iteration count
                    grads = gaussians.calculate_grad(query_filter_data)

                    # grad_dict = archive_grads(origin_xyz, query_filter_data, grads, grad_dict)
                    result_ddd = create_dict_from_single_grads(grads)

                    # TODO : reset point to original position
                    gaussians.reset_params()

                    # TODO : reset grads
                    gaussians.reset_grads()

                    #### save moved points
                    # red_color = [0, 255, 255]
                    # colors = [red_color for _ in range(len(list(new_xyz_dict.values())))]
                    #
                    # # 좌표를 PointCloud 객체로 변환
                    # point_cloud = o3d.geometry.PointCloud()
                    # point_cloud.points = o3d.utility.Vector3dVector(list(new_xyz_dict.values()))
                    # point_cloud.colors = o3d.utility.Vector3dVector(
                    #     np.asarray(colors) / 255.0)  # 0-255 RGB 값을 0-1 사이 값으로 변환
                    #
                    # # PLY 파일로 저장
                    # o3d.io.write_point_cloud("{}/moved_{}_point_cloud.ply".format(save_path, ii), point_cloud)
                    # # with open('{}/moved_xyz_{}.txt'.format(save_path, iteration), 'w') as file:
                    # #     for key, value in new_xyz_dict.items():
                    # #         file.write(f'{value}\n')
                    # print('new_xyz_SAVED')

                    # data save at 5
                    if ii == 7:
                        # print("*********")
                        # print("TXT SAVE*******")
                        # print('*********')
                        with open('{}/result_dict_{}_{}.txt'.format(save_path, random_v, iteration), 'w') as file:
                            for key, value in result_ddd.items():
                                file.write(f'{key}: {value}\n')
                        # print('result_dict_SAVED')

                        with open('{}/origin_xyz_{}.txt'.format(save_path, iteration), 'w') as file:
                            for key, value in origin_xyz_dict.items():
                                file.write(f'{key}: {value}\n')
                        # print('origin_xyz_SAVED')

                        with open('{}/mask_{}_{}.txt'.format(save_path, random_v, iteration), 'w') as file:
                            for key, value in dict_mask.items():
                                file.write(f'{key}: {value}\n')
                        # print('mask_SAVED')
                        # print('***********')
                        # print('SAVED PATH [RESULT DICT] : ', '{}/result_dict_{}_{}.txt'.format(save_path, random_v, iteration))
                        # print('SAVED PATH [origin_xyz] : ', '{}/origin_xyz_{}.txt'.format(save_path, iteration))
                        # print('SAVED PATH [mask] : ', '{}/mask_{}_{}.txt'.format(save_path, random_v, iteration))
                        # print('***********')
                        # print('***********')

            ii += 1

        elif iteration >= 7000 and iteration % 100 == 9:
            # print("@@@@@@@@@@@@@@@@@@@@")
            # print("@@@@ITERATION {}@@@@".format(iteration))
            # print("@@@@@@@@@@@@@@@@@@@@")
            # print("====================")
            # print("====================")
            # print("====XYZ SIZE : {}===".format(gaussians.get_xyz.shape[0]))
            # print("====================")
            # print("====================")
            # 6번째는 ply 저장 & update xyz
            xyz_len = gaussians.get_xyz.shape[0]
            # print('xyz len : ', xyz_len)
            # th_mask = make_th_mask(save_path, xyz_len, random_v, iteration)
            zero_mask = make_zero_mask(save_path, xyz_len, random_v, iteration)
            # print("########SAVING#########")
            # print("th_mask size: ", th_mask.size())
            # print("zero mask size : ", zero_mask.size())
            exp_avg_list, exp_avg_sq_list, ll3 = gaussians.save_optimizer()

            # 저장 1
            # # print("\n[ITER {}] Saving 1 Gaussians only th removed".format(iteration))
            # gaussians.filter_removed_points(th_mask)
            # scene.custom_save(iteration, 'only_th')
            # gaussians.reset_params()
            # gaussians.restore_optimizer(exp_avg_list, exp_avg_sq_list, ll3)

            # 저장 2
            # print("\n[ITER {}] Saving 2 Gaussians only zeros".format(iteration))
            if iteration % 1000 == 9:
                print("SAVING......")
                gaussians.filter_removed_points(zero_mask)
                scene.custom_save(iteration, 'only_zero')
                gaussians.reset_params()
                gaussians.restore_optimizer(exp_avg_list, exp_avg_sq_list, ll3)

                # 저장 3
                # print("\n[ITER {}] Saving 3 Gaussians filter by th".format(iteration))
                # gaussians.filter_points(th_mask)
                # scene.custom_save(iteration, 'filter_th')
                # gaussians.reset_params()
                # gaussians.restore_optimizer(exp_avg_list, exp_avg_sq_list, ll3)

                # 저장 4
                # print("\n[ITER {}] Saving 4 Gaussians filter by th".format(iteration))
                gaussians.filter_points(zero_mask)
                scene.custom_save(iteration, 'filter_zero')
                gaussians.reset_params()
                gaussians.restore_optimizer(exp_avg_list, exp_avg_sq_list, ll3)


            # zero인 애들 지우기
            gaussians.filter_points(zero_mask)
            print("Filtered Complete.")
            print("gaussian count : ", gaussians.get_xyz.shape[0])
            # print("_+_+_+_+_+_+_+_+_+_+_")
            # print("AFTER remove -> xyz SIZE : ", gaussians.get_xyz.size())

            ii += 1
            reset_result_dict()

        else:
            ii = 0
            # if iteration >= 7000 and (iteration - 7000) % 100 == 0:
                # print("9((((((9((((((9((((((")
                # print("NEW WWW xyz SIZE : ", gaussians.get_xyz.size())


            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


result_dict = {}


def create_dict_from_single_grads(grads):
    global result_dict  # 전역 변수를 함수 내부에서 사용하겠다고 선언

    for index, grad in enumerate(grads):
        grad_list = [item.item() for item in grad]
        if index in result_dict:
            result_dict[index].extend(grad_list)
        else:
            result_dict[index] = grad_list
    return result_dict


def reset_result_dict():
    global result_dict
    result_dict = {}


def tensor_to_dict(data_tensor):
    if not isinstance(data_tensor, torch.Tensor):
        raise ValueError("Input should be a PyTorch tensor.")

    data_dict = {index: row.tolist() for index, row in enumerate(data_tensor)}
    return data_dict


def archive_grads(data, indicator, grads, dicta):
    indices = torch.nonzero(indicator).squeeze()

    result_dict = dicta
    i = 0
    for index in indices:
        key = tuple(data[index].tolist())
        value = grads[index].item()
        if key in result_dict:
            result_dict[key]['grads'].append(value)
        else:
            result_dict[key] = {'grads': [value]}

    return result_dict


def random_query_filter(total_elements, num):
    total_size = total_elements.size()[0]
    # False로 이루어진 배열 생성
    data = np.full(total_size, False)

    # 랜덤하게 True로 설정
    true_indices = np.random.choice(total_size, num, replace=False)
    data[true_indices] = True

    # NumPy 배열을 PyTorch Tensor로 변환
    random_data = torch.tensor(data, dtype=torch.bool)
    return random_data


def query_filter(data, query):
    result = torch.any(torch.all(data.unsqueeze(1) == query.unsqueeze(0), dim=2), dim=1)
    return result


def select_random_pts(tensor_data, percent=0.05):
    num_elements = tensor_data.size(0)
    num_elements_to_select = int(percent * num_elements)

    # 랜덤하게 선택된 인덱스
    random_indices = torch.randperm(num_elements_to_select, device='cuda')

    # 랜덤하게 선택된 인덱스를 사용해 요소 추출
    selected_elements = tensor_data[random_indices]
    return selected_elements


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
