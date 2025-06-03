# import cv2 as cv
# import numpy as np
# from functools import partial
# import mmcv
# import mmengine
# from mmpose.apis.inferencers import MMPoseInferencer
# from mmpose.apis import (_track_by_iou, _track_by_oks,
#                          convert_keypoint_definition, extract_pose_sequence,
#                          inference_pose_lifter_model, inference_topdown,
#                          init_model)
# from mmpose.models.pose_estimators import PoseLifter
# from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import (PoseDataSample, merge_data_samples,
#                                split_instances)
# from mmpose.utils import adapt_mmdet_pipeline
# from mmdet.apis import inference_detector, init_detector

# def process_one_bbox (bbox_info, image, pose_estimator, pose_lifter, frame_idx, pose_est_results_last, pose_est_results_list, next_id):
#     pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
#     pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']
#     pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
#     pose_est_results_converted = []

#     # 进行棒球运动员的二维姿态估计
#     pose_est_results = inference_topdown(pose_estimator, image, bbox_info)

#     # 跟踪方法
#     _track = partial(_track_by_oks)

#     for i, data_sample in enumerate(pose_est_results):
#         pred_instances = data_sample.pred_instances.cpu().numpy()
#         keypoints = pred_instances.keypoints

#         # 计算面积和检测框
#         if 'bboxes' in pred_instances:
#             areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in pred_instances.bboxes])
#             pose_est_results[i].pred_instances.set_field(areas, 'areas')
#         else:
#             areas, bboxes = [], []
#             for keypoint in keypoints:
#                 xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
#                 xmax = np.max(keypoint[:, 0])
#                 ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
#                 ymax = np.max(keypoint[:, 1])
#                 areas.append((xmax - xmin) * (ymax - ymin))
#                 bboxes.append([xmin, ymin, xmax, ymax])
                
#             pose_est_results[i].pred_instances.areas = np.array(areas)
#             pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

#         track_id, pose_est_results_last, _ = _track(data_sample, pose_est_results_last, 0.3)
#         if track_id == -1:
#             if np.count_nonzero(keypoints[:, :, 1]) >= 3:
#                 track_id = next_id
#                 next_id += 1
#             else:
#                 # 如果检测到的关键点很少，则删除这个人
#                 keypoints[:, :, 1] = -10
#                 pose_est_results[i].pred_instances.set_field(keypoints, 'keypoints')
#                 pose_est_results[i].pred_instances.set_field(pred_instances.bboxes * 0, 'bboxes')
#                 pose_est_results[i].set_field(pred_instances, 'pred_instances')
#                 track_id = -1
#         pose_est_results[i].set_field(track_id, 'track_id')

#         # 添加关键点二维坐标
#         keypoints_2d = keypoints

#         # 转换关键点形式，用于三维提升
#         pose_est_result_converted = PoseDataSample()
#         pose_est_result_converted.set_field(pose_est_results[i].pred_instances.clone(), 'pred_instances')
#         pose_est_result_converted.set_field(pose_est_results[i].gt_instances.clone(), 'gt_instances')
#         keypoints = convert_keypoint_definition(keypoints, pose_det_dataset_name, pose_lift_dataset_name)
#         pose_est_result_converted.pred_instances.set_field(keypoints, 'keypoints')
#         pose_est_result_converted.set_field(pose_est_results[i].track_id, 'track_id')
#         pose_est_results_converted.append(pose_est_result_converted)

#     pose_est_results_list.append(pose_est_results_converted.copy())

#     # 提取并填充pose2d序列
#     pose_seq_2d = extract_pose_sequence(
#         pose_est_results_list,
#         frame_idx = frame_idx,
#         causal = pose_lift_dataset.get('causal', False),
#         seq_len = pose_lift_dataset.get('seq_len', 1),
#         step = pose_lift_dataset.get('seq_step', 1)
#     )

#     # 进行三维提升
#     pose_lift_results = inference_pose_lifter_model(
#         pose_lifter,
#         pose_seq_2d,
#         image_size = mmcv.bgr2rgb(image).shape[:2],
#         norm_pose_2d = True
#     )

#     # 后处理
#     for idx, pose_lift_result in enumerate(pose_lift_results):
#         pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

#         pred_instances = pose_lift_result.pred_instances
#         keypoints = pred_instances.keypoints
#         keypoint_scores = pred_instances.keypoint_scores
#         if keypoint_scores.ndim == 3:
#             keypoint_scores = np.squeeze(keypoint_scores, axis=1)
#             pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
#         if keypoints.ndim == 4:
#             keypoints = np.squeeze(keypoints, axis=1)

#         keypoints = keypoints[..., [0, 2, 1]]
#         keypoints[..., 0] = -keypoints[..., 0]
#         keypoints[..., 2] = -keypoints[..., 2]

#         pose_lift_results[idx].pred_instances.keypoints = keypoints

#         # 添加关键点三维坐标
#         keypoints_3d = keypoints

#     return keypoints_2d, keypoints_3d, pose_est_results, pose_est_results_list, next_id
    

# def get_baseball_man_keypoints (video, detector, pose_estimator, pose_lifter):
#     # 准备
#     baseball_man_kpt_2d_list = []
#     baseball_man_kpt_3d_list = []
#     frame_idx = 0
#     next_id_1 = 0
#     next_id_2 = 0
#     pose_est_results_list_1 = []
#     pose_est_results_1 = []
#     pose_est_results_list_2 = []
#     pose_est_results_2 = []

#     # 遍历视频
#     while video.isOpened():
#         frame_idx += 1
#         print(f"frame_idx: {frame_idx}")

#         # 读取视频，并判断是否读取成功
#         read_success, image = video.read()
#         if not read_success:
#             break

#         pose_est_results_last_1 = pose_est_results_1
#         pose_est_results_last_2 = pose_est_results_2

#         # 进行目标检测
#         det_result = inference_detector(detector, image)
#         pred_instance = det_result.pred_instances.cpu().numpy()

#         # 将人的检测框取出来
#         bboxes = pred_instance.bboxes
#         bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]

#         # 提取出棒球运动员的检测框
#         bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
#         sorted_idx = np.argsort(bboxes_area)

#         keypoints_2d_1, keypoints_3d_1, pose_est_results_1, pose_est_results_list_1, next_id_1 = process_one_bbox(
#             bboxes[sorted_idx[-1], :], 
#             image, 
#             pose_estimator, 
#             pose_lifter, 
#             frame_idx, 
#             pose_est_results_last_1, 
#             pose_est_results_list_1, 
#             next_id_1
#         )
#         baseball_man_kpt_2d_list.append(keypoints_2d_1)
#         baseball_man_kpt_3d_list.append(keypoints_3d_1)

#         keypoints_2d_2, keypoints_3d_2, pose_est_results_2, pose_est_results_list_2, next_id_2 = process_one_bbox(
#             bboxes[sorted_idx[-2], :], 
#             image, 
#             pose_estimator, 
#             pose_lifter, 
#             frame_idx, 
#             pose_est_results_last_2, 
#             pose_est_results_list_2, 
#             next_id_2
#         )
#         baseball_man_kpt_2d_list.append(keypoints_2d_2)
#         baseball_man_kpt_3d_list.append(keypoints_3d_2)

#     # 转化为numpy数组
#     baseball_man_kpt_2d = np.array(baseball_man_kpt_2d_list)
#     baseball_man_kpt_2d = baseball_man_kpt_2d.reshape((-1, 2, 17, 2))
#     baseball_man_kpt_3d = np.array(baseball_man_kpt_3d_list)
#     baseball_man_kpt_3d = baseball_man_kpt_3d.reshape((-1, 2, 17, 3))

#     return baseball_man_kpt_2d, baseball_man_kpt_3d

# def compute_displacement ():
#     pass

# def compute_velocity ():
#     pass

# def mark_data_to_image (displacement, velocity):
#     pass

# if __name__ == "__main__":
#     # 创建一个视频捕获器
#     video = cv.VideoCapture("baseball_pose.mp4")

#     # 获取视频相关数据
#     fps = video.get(cv.CAP_PROP_FPS)  # 帧率
#     width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))  # 宽度
#     height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # 高度

#     # 创建一个视频写入器
#     fourcc = cv.VideoWriter.fourcc('m', 'p', '4', 'v')
#     video_writer = cv.VideoWriter("baseball_pose_with_data.mp4", fourcc, fps, (width, height))

#     # 创建人类检测器
#     det_config = "mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
#     det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
#     detector = init_detector(det_config, det_checkpoint, device = "cuda:0")
#     detector.cfg = adapt_mmdet_pipeline(detector.cfg)

#     # 创建姿态估计器
#     est_config = "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
#     est_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
#     pose_estimator = init_model(est_config, est_checkpoint, device = "cuda:0")

#     # 创建维度提升器
#     lift_config = "mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"
#     lift_checkpoint = "https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth"
#     # lift_config = "mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py"
#     # lift_checkpoint = "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"
#     pose_lifter = init_model(lift_config, lift_checkpoint, device = "cuda:0")

#     # 获取关键点的二维像素坐标和三维坐标
#     baseball_man_kpt_2d, baseball_man_kpt_3d = get_baseball_man_keypoints(video, detector, pose_estimator, pose_lifter)

#     # 计算位移和速度，并标记到视频中
#     while video.isOpened():
#         # 读取视频，并判断是否读取成功
#         read_success, image = video.read()
#         if not read_success:
#             break



#     video.release()
#     video_writer.release()

import cv2 as cv
import numpy as np
from functools import partial
import mmcv
import mmengine
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector

def get_baseball_man_keypoints (video, detector, pose_estimator, pose_lifter):
    # 准备
    frame_idx = 0
    next_id = 0
    baseball_man_kpt_2d_list = []
    baseball_man_kpt_3d_list = []
    pose_est_results_list = []
    pose_est_results = []

    # 遍历视频
    while video.isOpened():
        frame_idx += 1
        print(f"frame_idx: {frame_idx}")

        # 读取视频，并判断是否读取成功
        read_success, image = video.read()
        if not read_success:
            break

        # 一些准备
        pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
        pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']
        pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
        pose_est_results_converted = []
        pose_est_results_last = pose_est_results

        # 进行目标检测
        det_result = inference_detector(detector, image)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # 将人的检测框取出来
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]

        # 提取出棒球运动员的检测框
        bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        sorted_idx = np.argsort(bboxes_area)
        baseball_man_bboxes = np.stack((bboxes[sorted_idx[-1], :], bboxes[sorted_idx[-2], :]), axis = 0)

        # 进行棒球运动员的二维姿态估计
        pose_est_results = inference_topdown(pose_estimator, image, baseball_man_bboxes)

        # 跟踪方法
        _track = partial(_track_by_oks)

        for i, data_sample in enumerate(pose_est_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints

            # 计算面积和检测框
            if 'bboxes' in pred_instances:
                areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in pred_instances.bboxes])
                pose_est_results[i].pred_instances.set_field(areas, 'areas')
            else:
                areas, bboxes = [], []
                for keypoint in keypoints:
                    xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                    xmax = np.max(keypoint[:, 0])
                    ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                    ymax = np.max(keypoint[:, 1])
                    areas.append((xmax - xmin) * (ymax - ymin))
                    bboxes.append([xmin, ymin, xmax, ymax])
                    
                pose_est_results[i].pred_instances.areas = np.array(areas)
                pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

            track_id, pose_est_results_last, _ = _track(data_sample, pose_est_results_last, 0.3)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = next_id
                    next_id += 1
                else:
                    # 如果检测到的关键点很少，则删除这个人
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances, 'pred_instances')
                    track_id = -1
            pose_est_results[i].set_field(track_id, 'track_id')

            # 添加关键点二维坐标
            baseball_man_kpt_2d_list.append(keypoints)

            # 转换关键点形式，用于三维提升
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(pose_est_results[i].pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.set_field(pose_est_results[i].gt_instances.clone(), 'gt_instances')
            keypoints = convert_keypoint_definition(keypoints, pose_det_dataset_name, pose_lift_dataset_name)
            pose_est_result_converted.pred_instances.set_field(keypoints, 'keypoints')
            pose_est_result_converted.set_field(pose_est_results[i].track_id, 'track_id')
            pose_est_results_converted.append(pose_est_result_converted)

        pose_est_results_list.append(pose_est_results_converted.copy())

        # 提取并填充pose2d序列
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx = frame_idx,
            causal = pose_lift_dataset.get('causal', False),
            seq_len = pose_lift_dataset.get('seq_len', 1),
            step = pose_lift_dataset.get('seq_step', 1)
        )

        # 进行三维提升
        pose_lift_results = inference_pose_lifter_model(
            pose_lifter,
            pose_seq_2d,
            image_size = mmcv.bgr2rgb(image).shape[:2],
            norm_pose_2d = True
        )

        # 后处理
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            pose_lift_results[idx].pred_instances.keypoints = keypoints

            # 添加关键点三维坐标
            baseball_man_kpt_3d_list.append(keypoints)

    # 转化为numpy数组
    baseball_man_kpt_2d = np.array(baseball_man_kpt_2d_list)
    baseball_man_kpt_2d = baseball_man_kpt_2d.reshape((-1, 2, 17, 2))
    baseball_man_kpt_3d = np.array(baseball_man_kpt_3d_list)
    baseball_man_kpt_3d = baseball_man_kpt_3d.reshape((-1, 2, 17, 3))

    return baseball_man_kpt_2d, baseball_man_kpt_3d

def compute_displacement ():
    pass

def compute_velocity ():
    pass

def mark_data_to_image (displacement, velocity):
    pass

if __name__ == "__main__":
    # 创建一个视频捕获器
    video = cv.VideoCapture("baseball_pose.mp4")

    # 获取视频相关数据
    fps = video.get(cv.CAP_PROP_FPS)  # 帧率
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # 高度

    # 创建一个视频写入器
    fourcc = cv.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_writer = cv.VideoWriter("baseball_pose_with_data.mp4", fourcc, fps, (width, height))

    # 创建人类检测器
    det_config = "mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    detector = init_detector(det_config, det_checkpoint, device = "cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 创建姿态估计器
    est_config = "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
    est_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
    pose_estimator = init_model(est_config, est_checkpoint, device = "cuda:0")

    # 创建维度提升器
    lift_config = "mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"
    lift_checkpoint = "https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth"
    # lift_config = "mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py"
    # lift_checkpoint = "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"
    pose_lifter = init_model(lift_config, lift_checkpoint, device = "cuda:0")

    # 获取关键点的二维像素坐标和三维坐标
    baseball_man_kpt_2d, baseball_man_kpt_3d = get_baseball_man_keypoints(video, detector, pose_estimator, pose_lifter)

    # 计算位移和速度，并标记到视频中
    while video.isOpened():
        # 读取视频，并判断是否读取成功
        read_success, image = video.read()
        if not read_success:
            break

    video.release()
    video_writer.release()