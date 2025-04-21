from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from .data_schema import ArgoverseScenario, ObjectState, ObjectType, Track


def translate(position, translation) -> None:
    """
    Applies a translation.
    :param x: <np.float: 3, 1>. Translation in x, y, z direction.
    """
    return (position + translation)

def rotate(position: np.ndarray, orientation: Quaternion, rotation_quaternion: Quaternion) -> None:
    """
    Rotates box.
    :param quaternion: Rotation to apply.
    """
    global_position = np.dot(rotation_quaternion.rotation_matrix, position)
    global_orientation = rotation_quaternion * orientation
    return global_position, global_orientation


def ego_to_global(position: np.ndarray, orientation: Quaternion, translation: np.ndarray, rotation_quaternion: Quaternion) -> np.ndarray:
    """
    Transforms a position from ego vehicle coordinates to global coordinates.

    Args:
        position (np.ndarray): A 3D vector representing the position in ego coordinates.
        translation (np.ndarray): A 3D vector representing the translation in global coordinates.
        rotation_quaternion (list[float]): A quaternion (qw, qx, qy, qz) representing the ego vehicle's rotation.

    Returns:
        np.ndarray: The position in global coordinates.
    """
    rotated_position, global_orientation = rotate(position, orientation, rotation_quaternion)
    global_position = translate(rotated_position, translation)
    return global_position, global_orientation


def convert_object_type(df: pd.DataFrame) -> pd.DataFrame:
    
    category_mapping = {
            'REGULAR_VEHICLE': 'vehicle',
            'LARGE_VEHICLE': 'vehicle',
            'BOX_TRUCK': 'vehicle',
            'TRUCK': 'vehicle',
            'SCHOOL_BUS': 'vehicle',
            'PEDESTRIAN': 'pedestrian',
            'MOTORCYCLIST': 'motorcyclist',
            'BICYCLIST': 'cyclist',
            'BUS': 'bus',
            'BOLLARD': 'static',
            'CONSTRUCTION_CONE': 'static',
            'CONSTRUCTION_BARREL': 'static',
            'STOP_SIGN': 'static',
            'SIGN': 'static',
            'MOBILE_PEDESTRIAN_SIGN': 'static',
            'MOBILE_PEDESTRIAN_CROSSING_SIGN': 'static',
            'BICYCLE': 'riderless_bicycle',
            'WHEELED_DEVICE': 'unknown',
            'MOTORCYCLE': 'unknown',
            'VEHICULAR_TRAILER': 'unknown',
            'TRUCK_CAB': 'unknown',
            'DOG': 'unknown',
            'WHEELED_RIDER': 'unknown',
            'STROLLER': 'unknown',
            'ARTICULATED_BUS': 'unknown',
            'MESSAGE_BOARD_TRAILER': 'unknown',
            'WHEELCHAIR': 'unknown',
            'RAILED_VEHICLE': 'unknown',
            'OFFICIAL_SIGNALER': 'unknown',
            'TRAFFIC_LIGHT_TRAILER': 'unknown',
            'ANIMAL': 'unknown'
        }
    
    df['category'] = df['category'].map(category_mapping)
    
    return df


def _load_tracks_from_tabular_format(tracks_df: pd.DataFrame) -> List[Track]:
    """Load tracks from tabular data format.

    Args:
        tracks_df: DataFrame containing all track data in a tabular format.

    Returns:
        All tracks associated with the scenario.
    """
    
    # track_id: 1.track_uuid
    # object_states: (timestep) 0.timestamp_ns, 13.timestep
    #                (position) 10.tx_m, 11.ty_m, 12.tz_m
    #                (heading) 6.qw, 7.qx, 8.qy, 9.qz, 
    #                (size) 3.length_m, 4.width_m, 5.height_m
    # object_type: 2.category
    # object_category: x
    tracks: List[Track] = []

    for track_id, track_df in tracks_df.groupby("track_uuid"):

        object_type: ObjectType = ObjectType(track_df["category"].iloc[0])
        
        timesteps: List[int] = track_df.loc[:, "timestep"].values.tolist()
        positions: List[Tuple[float, float]] = list(
            zip(
                track_df.loc[:, "tx_m"].values.tolist(),
                track_df.loc[:, "ty_m"].values.tolist(),
                track_df.loc[:, "tz_m"].values.tolist(),
            )
        )
        headings: List[Tuple[float, float, float, float]] = list(
            zip(
                track_df.loc[:, "qw"].values.tolist(),
                track_df.loc[:, "qx"].values.tolist(),
                track_df.loc[:, "qy"].values.tolist(),
                track_df.loc[:, "qz"].values.tolist(),
            )
        )
        sizes: List[Tuple[float, float, float]] = list(
            zip(
                track_df.loc[:, "length_m"].values.tolist(),
                track_df.loc[:, "width_m"].values.tolist(),
                track_df.loc[:, "height_m"].values.tolist(),
            )
        )

        object_states: List[ObjectState] = []
        for idx in range(len(timesteps)):
            object_states.append(
                ObjectState(
                    timestep=timesteps[idx],
                    position=positions[idx],
                    heading=headings[idx],
                    size=sizes[idx],
                )
            )

        tracks.append(
            Track(track_id=track_id, object_states=object_states, object_type=object_type)
        )

    return tracks

# function similar to 'load_argoverse_scenario_parquet'
def merge_scenario_and_ego(scenario_path, ego_data_path):
    
    ego_info = dict(
        track_uuid = 'AV',
        category = 'vehicle',
        length_m = 4.87, # Ford Hybrid Fusion
        width_m = 1.85,
        height_m = 1.48,
        num_interior_pts = 0
    )
    
    if not Path(scenario_path).exists():
        raise FileNotFoundError(f"No scenario exists at location: {scenario_path}.")
    tracks_df = pd.read_feather(scenario_path)
    ego_data = pd.read_feather(ego_data_path)
    
    tracks_df = convert_object_type(tracks_df)
    
    lidar_path = scenario_path.parent/'sensors'/'lidar' # code only to get timestamps
    feather_files = list(sorted(lidar_path.glob('*.feather')))
    lidar_timestamps = np.array([np.int64(file.stem) for file in feather_files], dtype=np.int64)
    ego_timestamps = np.array(ego_data['timestamp_ns'], dtype=np.int64)
    
    closest_indices = []
    
    for lidar_ts in lidar_timestamps: # 알고보니까 lidar_timestamp에 해당하는 timestamp가 ego정보에 다 포함이 되어 있더라고요. 그래서 사실 이렇게 코딩 안해도 됨
        differences = np.abs(ego_timestamps - lidar_ts)
        closest_index = np.argmin(differences)
        closest_indices.append(closest_index)
        
    closest_ego_data = ego_data.iloc[closest_indices].copy()
    
    assert closest_ego_data['timestamp_ns'].is_monotonic_increasing, "timestamps are not sorted in ascending order."
    assert np.all(lidar_timestamps[:-1] <= lidar_timestamps[1:]), "lidar_timestamps are not sorted in ascending order."
    
    # assert np.isin(tracks_df['timestamp_ns'].unique(), lidar_timestamps).all()
    if not np.isin(tracks_df['timestamp_ns'].unique(), lidar_timestamps).all(): # code for handling missing lidar data
        tracks_df = tracks_df[tracks_df['timestamp_ns'].isin(lidar_timestamps)]
        
    closest_ego_data['timestamp_ns'] = lidar_timestamps
    
    for key, value in ego_info.items():
        closest_ego_data[key] = value
        
    # new_order = [
    #     'timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m', 'height_m', 
    #     'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m', 'num_interior_pts'
    # ]
    # closest_ego_data = closest_ego_data[new_order]
    
    timestep_dict = {timestamp: index for index, timestamp in enumerate(lidar_timestamps)}
    tracks_df['timestep'] = tracks_df['timestamp_ns'].map(timestep_dict)
    closest_ego_data['timestep'] = closest_ego_data['timestamp_ns'].map(timestep_dict)
    
    
    # Convert Ego-vehicle reference system to Global coordinate system
    for _, ego_row in closest_ego_data.iterrows():
        ego_translation = np.array([ego_row['tx_m'], ego_row['ty_m'], ego_row['tz_m']])
        ego_rotation = Quaternion(ego_row['qw'], ego_row['qx'], ego_row['qy'], ego_row['qz'])

        timestep_tracks = tracks_df[tracks_df['timestep'] == ego_row['timestep']]

        for track_idx, track_row in timestep_tracks.iterrows():
            position = np.array([track_row['tx_m'], track_row['ty_m'], track_row['tz_m']])
            orientation = Quaternion(np.array(track_row[['qw', 'qx', 'qy', 'qz']]))

            global_position, global_orientation = ego_to_global(position, orientation, ego_translation, ego_rotation)

            tracks_df.loc[track_idx, ['tx_m', 'ty_m', 'tz_m']] = global_position
            tracks_df.loc[track_idx, ['qw', 'qx', 'qy', 'qz']] = [global_orientation.w, global_orientation.x, global_orientation.y, global_orientation.z]

    merged_tracks_df = pd.concat([tracks_df, closest_ego_data], ignore_index=True)
    assert merged_tracks_df['timestamp_ns'].nunique() == len(lidar_timestamps) # annotation's timestamp should be same as lidar's timestamp
    
    merged_tracks = _load_tracks_from_tabular_format(merged_tracks_df)
    
    scenario_id = scenario_path.parts[-2]
    return ArgoverseScenario(
        scenario_id=scenario_id,
        timestamps_ns=lidar_timestamps,
        tracks=merged_tracks,
    )