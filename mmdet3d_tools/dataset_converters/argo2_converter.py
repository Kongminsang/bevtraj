import os
from os import path as osp
import multiprocessing
from functools import partial

import mmengine
import numpy as np
import pandas as pd
from pyquaternion import Quaternion


Argo2NameMapping = {
    'REGULAR_VEHICLE': 'car',
    'PEDESTRIAN': 'pedestrian',
    'BOLLARD': 'barrier',
    'CONSTRUCTION_CONE': 'traffic_cone',
    'CONSTRUCTION_BARREL': 'barrier',
    'BICYCLE': 'bicycle',
    'LARGE_VEHICLE': 'construction_vehicle',
    'BUS': 'bus',
    'BOX_TRUCK': 'truck',
    'TRUCK': 'truck',
    'MOTORCYCLE': 'motorcycle',
    'BICYCLIST': 'bicycle',
    'VEHICULAR_TRAILER': 'trailer',
    'TRUCK_CAB': 'truck',
    'MOTORCYCLIST': 'motorcycle',
    'SCHOOL_BUS': 'bus',
    'WHEELED_RIDER': 'bicycle',
    'ARTICULATED_BUS': 'bus',
    'MESSAGE_BOARD_TRAILER': 'trailer',
    'MOBILE_PEDESTRIAN_SIGN': 'barrier',
    'MOBILE_PEDESTRIAN_CROSSING_SIGN': 'barrier',
    'WHEELCHAIR': 'pedestrian',
    'OFFICIAL_SIGNALER': 'pedestrian',
    'TRAFFIC_LIGHT_TRAILER': 'trailer'
}

categories_to_remove = [
    'STOP_SIGN',
    'SIGN',
    'DOG',
    'STROLLER',
    'RAILED_VEHICLE',
    'ANIMAL',
    'WHEELED_DEVICE'
]

def create_argo2_infos(root_path, info_prefix, max_sweeps=10, num_workers=8):
    """Create info file of argoverse2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
    """
    train_scene_tokens = os.listdir(root_path + '/train')
    val_scene_tokens = os.listdir(root_path + '/val')
    print('total train scene num: {}, total val scene num: {}'.format(
        len(train_scene_tokens), len(val_scene_tokens)))
    
    ### multiprocessing start
    
    print("...\nfilling infos...\n...")
    train_argo2_infos = run_fill_trainval_infos(root_path + '/train', train_scene_tokens, max_sweeps=max_sweeps, num_workers=num_workers)
    val_argo2_infos = run_fill_trainval_infos(root_path + '/val', val_scene_tokens, max_sweeps=max_sweeps, num_workers=num_workers)
    print('available train sample: {}, available val sample: {}'.format(
        len(train_argo2_infos), len(val_argo2_infos)))
    
    ### multiprocessing finished
    
    metadata = dict(version='av2-trainval')
    data = dict(infos=train_argo2_infos, metadata=metadata)
    info_train_path = osp.join(root_path,
                            '{}_infos_train.pkl'.format(info_prefix))
    mmengine.dump(data, info_train_path)
    
    data['infos'] = val_argo2_infos
    info_val_path = osp.join(root_path,
                                '{}_infos_val.pkl'.format(info_prefix))
    mmengine.dump(data, info_val_path)
    
    
def run_fill_trainval_infos(dataroot, scene_tokens, max_sweeps=10, num_workers=10):
    num_scenes = len(scene_tokens)
    if num_scenes < num_workers:
        print("Use one worker, as num_scenes < num_workers:")
        num_workers = 1
        
    argument_list = []
    num_scenes_each_worker = int(num_scenes // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            end_idx = num_scenes
        else:
            end_idx = (i + 1) * num_scenes_each_worker
        argument_list.append([scene_tokens[i * num_scenes_each_worker:end_idx], i])

    func = partial(
        fill_trainval_infos_wrapper,
        dataroot=dataroot,
        max_sweeps=max_sweeps)
    
    with multiprocessing.Pool(num_workers, maxtasksperchild=10) as p:
        ret = list(p.imap(func, argument_list))
        
    combined_infos = []
    for worker_result in ret:
        combined_infos.extend(worker_result)
        
    return combined_infos

    
def fill_trainval_infos_wrapper(
    args, dataroot, max_sweeps
):
    return _fill_trainval_infos(
        dataroot=dataroot,
        scene_tokens=args[0],
        max_sweeps=max_sweeps,
        worker_idx=args[1]
    )    


def _fill_trainval_infos(dataroot,
                         scene_tokens,
                         max_sweeps=10,
                         worker_idx=0):
    """Generate the train/val infos for available scenes from the raw data.

    Args:
        root_path (str): path of argo2 dataset.
        argo2 (:obj:`Argoverse2`): Dataset class in the Argoverse2 dataset.
        scene_tokens (list[str]): unique token of scenes.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    argo2_infos = []
    available_scene_num = 0 
    
    for scene_token in mmengine.track_iter_progress(scene_tokens):
        # data loading (especially timestamp)
        scene_path = osp.join(dataroot, scene_token)
        _annotations = pd.read_feather(osp.join(scene_path, 'annotations.feather'))
        annotations = _annotations[~_annotations['category'].isin(categories_to_remove)]
        anno_timestamps = sorted(annotations['timestamp_ns'].unique())
        
        lidar_root_path = osp.join(scene_path, 'sensors', 'lidar')
        lidar_timestamps = sorted([int(osp.splitext(file)[0]) for file in os.listdir(lidar_root_path) if file.endswith('.feather')])
        
        if not set(anno_timestamps) - set(lidar_timestamps): # Determine scene availability based on LiDAR file integrity
            available_scene_num += 1
            extrinsic_params = pd.read_feather(osp.join(scene_path, 'calibration', 'egovehicle_SE3_sensor.feather'))
            intrinsic_params = pd.read_feather(osp.join(scene_path, 'calibration', 'intrinsics.feather'))
            
            # extracting ego_pose for lidar timestamps, transform matrix
            ego_pose = pd.read_feather(osp.join(scene_path, 'city_SE3_egovehicle.feather'))
            
            # lidar extrinsic parameters
            up_l2e_t = extrinsic_params[extrinsic_params['sensor_name'] == 'up_lidar'][['tx_m', 'ty_m', 'tz_m']].iloc[0].values.tolist()
            up_l2e_r = extrinsic_params[extrinsic_params['sensor_name'] == 'up_lidar'][['qw', 'qx', 'qy', 'qz']].iloc[0].values.tolist()
            up_l2e_r_mat = Quaternion(up_l2e_r).rotation_matrix
            
            # extract each camera's timestamp
            ring_cam_types = [ # we only use mono cams by 10Hz
                'ring_front_center',
                'ring_front_left',
                'ring_front_right',
                'ring_rear_left',
                'ring_rear_right',
                'ring_side_left',
                'ring_side_right',
            ]
            ring_cam_timestamps = dict()
            for cam in ring_cam_types:
                cam_path = osp.join(scene_path, 'sensors', 'cameras', cam)
                cam_timestamps = sorted([int(osp.splitext(file)[0]) for file in os.listdir(cam_path) if file.endswith('.jpg')])
                ring_cam_timestamps.update({cam:cam_timestamps})
            
            # timestamps dictionary for calculating box_velocity
            each_anno_timestamps = {}
            for track_uuid in annotations['track_uuid'].unique():
                each_annotations = annotations[annotations['track_uuid'] == track_uuid]
                each_anno_timestamps.update({track_uuid: sorted(each_annotations['timestamp_ns'].unique())})
            
            # append each frame's info dictionary
            for i, timestamp in enumerate(lidar_timestamps):
                lidar_path = osp.join(lidar_root_path, f'{timestamp}.feather')
                mmengine.check_file_exist(lidar_path)
                
                try:
                    ego2global = ego_pose[ego_pose['timestamp_ns'] == timestamp].iloc[0]
                except:
                    raise ValueError("ego_pose_timestamp does not include the lidar timestamp.")
                e2g_t = ego2global[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                e2g_r = ego2global[['qw', 'qx', 'qy' ,'qz']].values.tolist()
                e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                
                info = {
                    'lidar_path': lidar_path,
                    'num_features': 5,
                    'scene_token': scene_token,
                    'token': scene_token + '.' + str(timestamp),
                    'sweeps': [],
                    'cams': dict(),
                    'lidar2ego_translation': up_l2e_t,
                    'lidar2ego_rotation': up_l2e_r,
                    'ego2global_translation': e2g_t,
                    'ego2global_rotation': e2g_r,
                    'timestamp': timestamp,
                }
                
                # cam infos
                for cam in ring_cam_types:
                    # cam timestamp (closest to lidar timestamp)
                    cam_timestamps = np.array(ring_cam_timestamps[cam], dtype=np.int64)
                    c2l_differences = np.abs(cam_timestamps - timestamp)
                    c2l_closest_index = np.argmin(c2l_differences)
                    cam_timestamp = cam_timestamps[c2l_closest_index]
                    
                    cam_path = osp.join(scene_path, 'sensors', 'cameras', cam, f'{cam_timestamp}.jpg')
                    
                    # cam_extrinsic
                    cam_extrinsic = extrinsic_params[extrinsic_params['sensor_name'] == cam].iloc[0]
                    c2e_t = cam_extrinsic[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                    c2e_r = cam_extrinsic[['qw', 'qx', 'qy', 'qz']].values.tolist()
                    
                    # ego_pose at cam_timestamp
                    try:
                        ego_pose_cam = ego_pose[ego_pose['timestamp_ns'] == cam_timestamp].iloc[0]
                    except:
                        raise ValueError("ego_pose_timestamp does not include the camera timestamp.")
                    e2g_t_c = ego_pose_cam[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                    e2g_r_c = ego_pose_cam[['qw', 'qx', 'qy', 'qz']].values.tolist()
                    
                    # cam_intrinsic
                    cam_intrinsic = intrinsic_params[intrinsic_params['sensor_name'] == cam].iloc[0]
                    cam_intrinsic_mat = np.array([
                        [cam_intrinsic['fx_px'], 0, cam_intrinsic['cx_px']],
                        [0, cam_intrinsic['fy_px'], cam_intrinsic['cy_px']],
                        [0, 0, 1]
                    ])

                    cam_info = obtain_sensor2top(c2e_t, 
                                                 c2e_r,
                                                 e2g_t_c,
                                                 e2g_r_c,
                                                 e2g_t,
                                                 e2g_r_mat,
                                                 up_l2e_t,
                                                 up_l2e_r_mat)
                    cam_info.update(data_path=cam_path,
                                    type=cam,
                                    sample_data_token=scene_token+'.'+str(cam_timestamp),
                                    timestamp=cam_timestamp,
                                    cam_intrinsic=cam_intrinsic_mat)
                    info['cams'].update({cam: cam_info})
                    
                    
                # obtain lidar sweeps for a single key-frame
                if i > max_sweeps: 
                    lidar_sweep_timestamps = lidar_timestamps[i-max_sweeps:i]
                else:
                    lidar_sweep_timestamps = lidar_timestamps[:i]
                lidar_sweep_timestamps.sort(reverse=True)
                
                sweeps = []
                for lidar_sweep_timestamp in lidar_sweep_timestamps:
                    ego2global_s = ego_pose[ego_pose['timestamp_ns'] == lidar_sweep_timestamp].iloc[0]
                    e2g_t_s = ego2global_s[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                    e2g_r_s = ego2global_s[['qw', 'qx', 'qy' ,'qz']].values.tolist()
                    sweep = obtain_ego2top(e2g_t_s, 
                                           e2g_r_s, 
                                           e2g_t,
                                           e2g_r_mat,
                                           up_l2e_t,
                                           up_l2e_r_mat)
                    sweep.update(data_path=osp.join(lidar_root_path, f'{lidar_sweep_timestamp}.feather'),
                                 type='lidar',
                                 sample_data_token=scene_token+'.'+str(lidar_sweep_timestamp),
                                 timestamp=lidar_sweep_timestamp,
                                 sensor2ego_translation=up_l2e_t,
                                 sensor2ego_rotation=up_l2e_r)
                    sweeps.append(sweep)
                info['sweeps'] = sweeps
                
                # obtain annotation
                curr_annotations = annotations[annotations['timestamp_ns'] == timestamp]
                box_list = curr_annotations.to_dict(orient='records')
                
                velocities = []
                rotations = []
                locations = []
                for box in box_list:
                    locs = np.array([box['tx_m'], box['ty_m'], box['tz_m']])
                    locs -= up_l2e_t
                    locs = Quaternion(up_l2e_r).inverse.rotation_matrix @ locs
                    
                    rots = Quaternion([box['qw'], box['qx'], box['qy'], box['qz']])
                    rots = (Quaternion(up_l2e_r).inverse * rots).yaw_pitch_roll[0]
                    
                    # Calculate velocity in a manner similar to the NuScenes dataset.
                    box_id = box['track_uuid']
                    box_anno = annotations[annotations['track_uuid'] == box_id]
                    box_timestamps = each_anno_timestamps[box_id]
                    idx = box_timestamps.index(timestamp)
                    
                    if idx > 4:
                        pos_first_timestamp = box_timestamps[idx-5]
                    else: 
                        pos_first_timestamp = box_timestamps[0]
                    if idx < len(box_timestamps)-5:
                        pos_last_timestamp = box_timestamps[idx+5]
                    else: 
                        pos_last_timestamp = box_timestamps[-1]
                    
                    loc_first = np.array(box_anno[box_anno['timestamp_ns'] == pos_first_timestamp].iloc[0][['tx_m', 'ty_m', 'tz_m']])
                    loc_last = np.array(box_anno[box_anno['timestamp_ns'] == pos_last_timestamp].iloc[0][['tx_m', 'ty_m', 'tz_m']])
                    
                    ego_pose_first = ego_pose[ego_pose['timestamp_ns'] == pos_first_timestamp].iloc[0]
                    e2g_t_first = ego_pose_first[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                    e2g_r_first = ego_pose_first[['qw', 'qx', 'qy' ,'qz']].values.tolist()
                    e2g_r_first_mat = Quaternion(e2g_r_first).rotation_matrix
                    
                    ego_pose_last = ego_pose[ego_pose['timestamp_ns'] == pos_last_timestamp].iloc[0]
                    e2g_t_last = ego_pose_last[['tx_m', 'ty_m', 'tz_m']].values.tolist()
                    e2g_r_last = ego_pose_last[['qw', 'qx', 'qy' ,'qz']].values.tolist()
                    e2g_r_last_mat = Quaternion(e2g_r_last).rotation_matrix
                    
                    if len(str(pos_first_timestamp)) == len(str(pos_last_timestamp)) == 18:
                        time_diff = (pos_last_timestamp - pos_first_timestamp) * 1e-9
                    else:
                        raise ValueError("len of timestamps should be 18")
                    velo = box_velocity(loc_first,
                                        loc_last,
                                        time_diff,
                                        e2g_t_first,
                                        e2g_r_first_mat,
                                        e2g_t_last,
                                        e2g_r_last_mat,
                                        e2g_r_mat,
                                        up_l2e_r_mat)
                    locations.append(locs)
                    rotations.append(rots)
                    velocities.append(velo)
                    
                locations = np.array(locations).reshape(-1, 3)
                rotations = np.array(rotations).reshape(-1, 1)
                velocities = np.array(velocities).reshape(-1, 2)
                
                dims = np.array([[b['width_m'], b['length_m'], b['height_m']] for b in box_list]).reshape(-1, 3)
                # we need to convert box size to
                # the format of our lidar coordinate system
                # which is x_size, y_size, z_size (corresponding to l, w, h)
                gt_boxes = np.concatenate([locations, dims[:, [1, 0, 2]], rotations], axis=1)
                assert len(gt_boxes) == len(box_list), f'gt_boxes: {len(gt_boxes)}, box_list: {len(box_list)}'
                
                names = [b['category'] for b in box_list]
                for j in range(len(names)):
                    if names[j] in Argo2NameMapping:
                        names[j] = Argo2NameMapping[names[j]]
                    else:
                        if not names[j] in categories_to_remove:
                            print("unknown category: ", names[j])
                names = np.array(names)
                
                valid_flag = np.array([b['num_interior_pts'] > 0 for b in box_list], dtype=bool).reshape(-1)
                
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_velocity'] = velocities
                info['num_lidar_pts'] = np.array([b['num_interior_pts'] for b in box_list])
                info['valid_flag'] = valid_flag
                
                argo2_infos.append(info)
        
    print("available scene num: ", available_scene_num)
    return argo2_infos


def obtain_sensor2top(s2e_t_s, # 's' means sweep or other sensor(ex. camera)
                      s2e_r_s,
                      e2g_t_s, 
                      e2g_r_s,
                      e2g_t,
                      e2g_r_mat,
                      up_l2e_t, # top = up
                      up_l2e_r_mat):
    s2e_r_s_mat = Quaternion(s2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    
    # obtain the RT from sensor to Top LiDAR
    # sweep -> ego -> global -> ego' -> lidar
    R = (s2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T)
    T = (s2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T
                  ) + up_l2e_t @ np.linalg.inv(up_l2e_r_mat).T
    # points @ R.T + T
    
    info = {
        'sensor2ego_translation': s2e_t_s, # cam
        'sensor2ego_rotation': s2e_r_s, # cam
        'ego2global_translation': e2g_t_s, # cam_timestamp
        'ego2global_rotation': e2g_r_s, # cam_timestamp
        'sensor2lidar_translation': T, # cam -> up_lidar
        'sensor2lidar_rotation': R.T, # cam -> up_lidar
    }
    return info


def obtain_ego2top(e2g_t_s,
                   e2g_r_s,
                   e2g_t,
                   e2g_r_mat,
                   up_l2e_t,
                   up_l2e_r_mat):
    """
    The LiDAR point cloud data in the Argoverse 2 dataset is provided in the ego-vehicle coordinate system.
    To minimize discrepancies when working with data from the NuScenes dataset, which provides point clouds
    in the sensor (up_lidar) coordinate system, we convert the Argoverse 2 point cloud data to the up_lidar
    coordinate system.
    This function calculates the transformation matrix to align the lidar_sweep data with the current frame's
    up_lidar coordinate system.
    """
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    
    # function only for lidar_sweep (ego-vehicle reference frame)
    # obtain the RT from ego to Top LiDAR
    # ego(past) -> global -> ego(current) -> lidar
    R = e2g_r_s_mat.T @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T
    T = e2g_t_s @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T
                  ) + up_l2e_t @ np.linalg.inv(up_l2e_r_mat).T
    # points @ R.T + T
    
    info = {
        'ego2global_translation': e2g_t_s, # lidar_sweep timestamp
        'ego2global_rotation': e2g_r_s, # lidar_sweep timestamp
        'ego2lidar_translation': T, # ego(past) -> up_lidar
        'ego2lidar_rotation': R.T, # ego(past) -> up_lidar
    }
    return info


def box_velocity(loc_first,
                 loc_last,
                 time_diff,
                 e2g_t_first,
                 e2g_r_first_mat,
                 e2g_t_last,
                 e2g_r_last_mat,
                 e2g_r_mat,
                 up_l2e_r_mat):
    """
    param1: pos_first in ego-vehicle reference frame
    param2: pos_last in ego-vehicle reference frame
    param3: time_diff
    
    param4: ego2global translation for pos_first
    param5: ego2global rotation for pos_first
    param6: ego2global translation for pos_last
    param7: ego2global rotation for pos_last
    
    param8: ego2global rotation for current frame
    param9: up_lidar2ego rotation for current frame
    """
    loc_first = e2g_r_first_mat @ loc_first + e2g_t_first
    loc_last = e2g_r_last_mat @ loc_last + e2g_t_last
    
    if time_diff == 0:
        return np.array([np.nan, np.nan])
    if np.array_equal(loc_first, loc_last):
        return np.array([np.nan, np.nan])
    velo = ((loc_last - loc_first) / time_diff)
    velo = np.array([*velo[:2], 0.0])
    
    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(up_l2e_r_mat).T
    
    return velo[:2] # only x, y velocity
    