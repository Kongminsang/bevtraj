import os
import os.path as osp
import logging

import tqdm

from md_scenarionet.converter.utils import mph_to_kmh
import math
import geopandas as gpd
from shapely.ops import unary_union
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

from .scenario_serialization import merge_scenario_and_ego

from md_scenarionet.converter.argoverse2.type import get_traffic_obj_type, get_lane_type, get_lane_mark_type
from av2.datasets.motion_forecasting import scenario_serialization

from av2.map.map_api import ArgoverseStaticMap
from typing import Final, Tuple
from shapely.geometry import Point, Polygon

# _ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
# _ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
# _ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
# _ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
# _ESTIMATED_PEDESTRIAN_LENGTH_M: Final[float] = 0.5
# _ESTIMATED_PEDESTRIAN_WIDTH_M: Final[float] = 0.5
# _ESTIMATED_BUS_LENGTH_M: Final[float] = 12.0
# _ESTIMATED_BUS_WIDTH_M: Final[float] = 2.5

_HIGHWAY_SPEED_LIMIT_MPH: Final[float] = 85.0


def extract_tracks(tracks, track_length, sdc_idx='AV'):
    ret = dict()

    def _object_state_template(object_id):
        return dict(type=None, state=dict(# Never add extra dim if the value is scalar.
            position=np.zeros([track_length, 3], dtype=np.float32), length=np.zeros([track_length], dtype=np.float32),
            width=np.zeros([track_length], dtype=np.float32), height=np.zeros([track_length], dtype=np.float32),
            heading=np.zeros([track_length], dtype=np.float32), velocity=np.zeros([track_length, 2], dtype=np.float32),
            valid=np.zeros([track_length], dtype=bool), ),
            metadata=dict(track_length=track_length, type=None, object_id=object_id, dataset="av2"))

    def _heading_and_velocity(heading_quat, pre_position, position): # Universal regardless of coordinate system
        qw, qx, qy, qz = heading_quat
        pre_tx, pre_ty = pre_position[:2]
        tx, ty = position[:2]
        
        heading_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)) # yaw
        vx = (tx - pre_tx) / 0.1
        vy = (ty - pre_ty) / 0.1
        
        return heading_yaw, vx ,vy
        
        
    # track_category = []

    for obj in tracks:
        object_id = obj.track_id
        # track_category.append(obj.category.value)
        obj_state = _object_state_template(object_id)
        # Transform it to Waymo type string
        obj_state["type"] = get_traffic_obj_type(obj.object_type)

        length = obj.object_states[0].size[0]
        width = obj.object_states[0].size[1]
        height = obj.object_states[0].size[2]

        pre_position: Tuple[float, float] = (0.0, 0.0)
        for i, state in enumerate(obj.object_states):
            step_count = int(state.timestep)
            if i == 0:
                first_step_count = step_count
            if i == 1:
                second_step_count = step_count
            obj_state["state"]["position"][step_count][0] = state.position[0]
            obj_state["state"]["position"][step_count][1] = state.position[1]
            obj_state["state"]["position"][step_count][2] = state.position[2]

            obj_state["state"]["length"][step_count] = length
            obj_state["state"]["width"][step_count] = width
            obj_state["state"]["height"][step_count] = height

            heading, vx, vy = _heading_and_velocity(state.heading, pre_position, state.position)
            pre_position = state.position
            obj_state["state"]["heading"][step_count] = heading

            obj_state["state"]["velocity"][step_count][0] = vx
            obj_state["state"]["velocity"][step_count][1] = vy

            obj_state["state"]["valid"][step_count] = True

        obj_state["metadata"]["type"] = obj_state["type"]
        try:
            obj_state["state"]["velocity"][first_step_count][:] = obj_state["state"]["velocity"][second_step_count][:]
        except:
            obj_state["state"]["velocity"][first_step_count][:] = np.array([0.0, 0.0], dtype=np.float32)
        ret[object_id] = obj_state

    return ret


def extract_lane_mark(lane_mark):
    line = dict()
    line["type"] = get_lane_mark_type(lane_mark.mark_type)
    line["polyline"] = lane_mark.polyline.astype(np.float32)
    return line


def extract_map_features(map_features):
    ret = {}
    vector_lane_segments = map_features.get_scenario_lane_segments()
    vector_drivable_areas = map_features.get_scenario_vector_drivable_areas()
    ped_crossings = map_features.get_scenario_ped_crossings()

    ids = map_features.get_scenario_lane_segment_ids()

    max_id = max(ids)
    for seg in vector_lane_segments:
        center = {}
        lane_id = str(seg.id)

        left_id = str(seg.id + max_id + 1)
        right_id = str(seg.id + max_id + 2)
        left_marking = extract_lane_mark(seg.left_lane_marking)
        right_marking = extract_lane_mark(seg.right_lane_marking)

        ret[left_id] = left_marking
        ret[right_id] = right_marking

        center["speed_limit_mph"] = _HIGHWAY_SPEED_LIMIT_MPH

        center["speed_limit_kmh"] = mph_to_kmh(_HIGHWAY_SPEED_LIMIT_MPH)

        center["type"] = get_lane_type(seg.lane_type)

        polyline = map_features.get_lane_segment_centerline(seg.id)
        center["polyline"] = polyline.astype(np.float32)

        center["interpolating"] = True

        center["entry_lanes"] = [str(id) for id in seg.predecessors]

        center["exit_lanes"] = [str(id) for id in seg.successors]

        center["left_boundaries"] = []

        center["right_boundaries"] = []

        center["left_neighbor"] = []

        center["right_neighbor"] = []
        center['width'] = np.zeros([len(polyline), 2], dtype=np.float32)

        ret[lane_id] = center

    polygons = []
    for polygon in vector_drivable_areas:
        # convert to shapely polygon
        points = polygon.area_boundary
        polygons.append(Polygon([(p.x, p.y) for p in points]))

    polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
    boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        for i in range(0, len(block_points), 20):
            id = f'boundary_{idx}{i}'
            ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points[i:i + 20]}

    for cross in ped_crossings:
        bound = dict()
        bound["type"] = MetaDriveType.CROSSWALK
        bound["polygon"] = cross.polygon.astype(np.float32)
        ret[str(cross.id)] = bound

    return ret


def get_av2_sensor_scenarios(av2_data_directory, start_index, num):
    # parse raw data from input path to output path
    # annotation can be trajectory in sensor dataset
    logger.info("\nReading raw data")

    all_scenario_files = sorted(Path(av2_data_directory).rglob("annotations.feather"))
    
    return all_scenario_files
            

def convert_av2_sensor_scenario(scenario, version):
    md_scenario = SD()

    md_scenario[SD.ID] = scenario.scenario_id
    md_scenario[SD.VERSION] = version

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.
    track_length = scenario.timestamps_ns.shape[0]

    tracks = extract_tracks(scenario.tracks, track_length)

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario.static_map)
    md_scenario[SD.MAP_FEATURES] = map_features

    # compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA]["timestamp_ns"] = scenario.timestamps_ns
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = 'AV'
    md_scenario[SD.METADATA]["dataset"] = "av2"
    md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id
    md_scenario[SD.METADATA]["source_file"] = scenario.scenario_id
    md_scenario[SD.METADATA]["track_length"] = track_length

    # obj id
    obj_keys = list(tracks.keys())
    # md_scenario[SD.METADATA]["objects_of_interest"] = [obj_keys[idx] for idx, cat in enumerate(category) if cat == 2] 
    md_scenario[SD.METADATA]["objects_of_interest"] = [] 

    md_scenario[SD.METADATA]["sdc_track_index"] = obj_keys.index('AV')

    # clean memory
    del scenario
    return md_scenario

def is_scenario_available(scene_path):
    annotations = pd.read_feather(scene_path)
    anno_timestamps = sorted(annotations['timestamp_ns'].unique())

    lidar_root_path = osp.join(scene_path.parents[0], 'sensors', 'lidar')
    lidar_timestamps = sorted([int(osp.splitext(file)[0]) for file in os.listdir(lidar_root_path) if file.endswith('.feather')])
    return set(anno_timestamps) - set(lidar_timestamps)
    

def preprocess_av2_sensor_scenarios(files, worker_index): # generator function

    for scenario_path in tqdm(files, desc="Process av2_sensor scenarios for worker {}".format(worker_index)):
        non_available = is_scenario_available(scenario_path)
        if non_available:
            print("non-available scene: ", scenario_path.parents[0].name)
            continue
        ego_track_path = (scenario_path.parents[0] / "city_SE3_egovehicle.feather")
        scenario_merged = merge_scenario_and_ego(scenario_path, ego_track_path)
        
        static_map_dir = (scenario_path.parents[0] / "map")
        static_map_path = next(static_map_dir.glob("log_map_archive_*.json"), None)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        static_map.log_id = scenario_path.parents[0].name
        scenario_merged.static_map = static_map
        
        yield scenario_merged