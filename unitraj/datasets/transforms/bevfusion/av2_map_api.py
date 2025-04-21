import numpy as np
import cv2
from pathlib import Path
import json

from typing import Dict, List, Tuple, Optional, Union
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box


Geometry = Union[Polygon, LineString]

SOLID_LINE = ['SOLID_WHITE', 'SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_SOLID_YELLOW', 'SOLID_BLUE', 'NONE', 'UNKNOWN']
DASHED_LINE = ['DASHED_WHITE', 'DASHED_YELLOW', 'DOUBLE_DASH_WHITE', 'DOUBLE_DASH_YELLOW']
SOLID_DASH_LINE = ['SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW']
DASH_SOLID_LINE = ['DASH_SOLID_WHITE', 'DASH_SOLID_YELLOW']


class ArgoMapExplorer:
    def __init__(self, data_root):
        self.maps = {}
        all_map_paths = Path(data_root).rglob("log_map_archive_*.json")
        for map_path in all_map_paths:
            self.maps[map_path.parts[-3]] = json.load(open(map_path, "r"))
            
    def get_map_mask(self,
                     scene_token: str,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     canvas_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param scene_token: Unique identifier of the scene.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        map_geom = self.get_map_geom(scene_token, patch_box, patch_angle)

        # Convert geometry of each layer into mask and stack them into a numpy tensor.
        # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        map_mask = self.map_geom_to_mask(map_geom, local_box, canvas_size)
        assert np.all(map_mask.shape[1:] == canvas_size)

        return map_mask
    
    def get_map_geom(self,
                     scene_token: str,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float) -> List[Tuple[str, List[Geometry]]]:
        map_data = self.maps[scene_token]
        patch = self.get_patch_coord(patch_box, patch_angle)
        
        drivable_areas = self._get_layer_polygon(patch, patch_box, patch_angle,
                                                 map_data['drivable_areas'], layer_name='drivable_area')
        ped_crossings = self._get_layer_polygon(patch, patch_box, patch_angle,
                                                map_data['pedestrian_crossings'], layer_name='ped_crossing')
        
        map_geom = [("drivable_area", drivable_areas),
                    ("ped_crossing", ped_crossings)]
        
        lane_geom = self._get_lane_geom(patch, patch_box, patch_angle, map_data['lane_segments'])
        map_geom.extend(lane_geom)
        
        return map_geom

    def map_geom_to_mask(self,
                         map_geom: List[Tuple[str, List[Geometry]]],
                         local_box: Tuple[float, float, float, float],
                         canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param map_geom: List of layer names and their corresponding geometries.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        :return: Stacked numpy array of size [c x h x w] with c channels and the same height/width as the canvas.
        """
        # Get each layer mask and stack them into a numpy tensor.
        map_mask = []
        for layer_name, layer_geom in map_geom:
            layer_mask = self._layer_geom_to_mask(layer_name, layer_geom, local_box, canvas_size)
            if layer_mask is not None:
                map_mask.append(layer_mask)

        return np.array(map_mask)
    
    
    def _get_layer_polygon(self,
                           patch: Polygon,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           map_data: Dict[str, dict],
                           layer_name: str) -> List[Polygon]:
        patch_x, patch_y = patch_box[0], patch_box[1]
        
        polygon_list = []
        for element in map_data.values():
            if layer_name == 'drivable_area':
                exterior_coords = [(p['x'], p['y']) for p in element['area_boundary']]
            elif layer_name == 'ped_crossing':
                exterior_coords = [(p['x'], p['y']) for p in element['edge1'] + element['edge2'][::-1]]
            else: raise ValueError("Invalid layer name")
            
            polygon = Polygon(exterior_coords).intersection(patch)
            if not polygon.is_empty:
                polygon = affinity.rotate(polygon, -patch_angle, origin=(patch_x, patch_y))
                polygon = affinity.affine_transform(polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                polygon_list.append(polygon)
        
        return polygon_list
        
    def _get_lane_geom(self,
                        patch: Polygon,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        lane_segments: Dict[str, dict]) -> Optional[List[LineString]]:
        patch_x, patch_y = patch_box[0], patch_box[1]
        
        solid_lines, dashed_lines, solid_dash_lines, dash_solid_lines = [], [], [], []
        lane_type_to_list = {
            **{lt: solid_lines for lt in SOLID_LINE},
            **{lt: dashed_lines for lt in DASHED_LINE},
            **{lt: solid_dash_lines for lt in SOLID_DASH_LINE},
            **{lt: dash_solid_lines for lt in DASH_SOLID_LINE},
        }
        
        for lane_segment in lane_segments.values():
            if lane_segment['is_intersection']:
                continue
            for side in ('left', 'right'):
                line_nodes = [(p['x'], p['y']) for p in lane_segment[f'{side}_lane_boundary']]
                line = LineString(line_nodes).intersection(patch)
                if not line.is_empty:
                    line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y))
                    line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    
                    lane_type = lane_segment[f'{side}_lane_mark_type']
                    lane_type_to_list[lane_type].append(line)
                    
        lane_geom = [('solid_lines', solid_lines),
                    ('dashed_lines', dashed_lines),
                    ('solid_dash_lines', solid_dash_lines),
                    ('dash_solid_lines', dash_solid_lines)]
        return lane_geom

    def _layer_geom_to_mask(self,
                            layer_name: str,
                            layer_geom: List[Geometry],
                            local_box: Tuple[float, float, float, float],
                            canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Wrapper method that gets the mask for each layer's geometries.
        :param layer_name: The name of the layer for which we get the masks.
        :param layer_geom: List of the geometries of the layer specified in layer_name.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        """
        if layer_name in ['ped_crossing', 'drivable_area']:
            return self._polygon_geom_to_mask(layer_geom, local_box, canvas_size)
        elif layer_name in ['solid_lines', 'dashed_lines', 'solid_dash_lines', 'dash_solid_lines']:
            return self._line_geom_to_mask(layer_geom, local_box, canvas_size)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))
        
    def _polygon_geom_to_mask(self,
                              layer_geom: List[Polygon],
                              local_box: Tuple[float, float, float, float],
                              canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert polygon inside patch to binary mask and return the map patch.
        :param layer_geom: list of polygons for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch with the size canvas_size.
        """
        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]

        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        for polygon in layer_geom:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.affine_transform(new_polygon,
                                                        [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_polygon = affinity.scale(new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0))
                
                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                map_mask = self.mask_for_polygons(new_polygon, map_mask)

        return map_mask

    def _line_geom_to_mask(self,
                           layer_geom: List[LineString],
                           local_box: Tuple[float, float, float, float],
                           canvas_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Convert line inside patch to binary mask and return the map patch.
        :param layer_geom: list of LineStrings for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch in a canvas size.
        """
        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h/patch_h
        scale_width = canvas_w/patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        for line in layer_geom:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                map_mask = self.mask_for_lines(new_line, map_mask)
        return map_mask
    
    
    @staticmethod
    def get_patch_coord(patch_box: Tuple[float, float, float, float],
                        patch_angle: float = 0.0) -> Polygon:
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch
    
    @staticmethod
    def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
        """
        Convert a polygon or multipolygon list to an image mask ndarray.
        :param polygons: List of Shapely polygons to be converted to numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray polygon mask.
        """
        if not polygons:
            return mask

        def int_coords(x):
            # function to round and convert to int
            return np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        # interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        # cv2.fillPoly(mask, interiors, 0)
        return mask

    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        if lines.geom_type == 'MultiLineString':
            for line in lines:
                coords = np.asarray(list(line.coords), np.int32)
                coords = coords.reshape((-1, 2))
                cv2.polylines(mask, [coords], False, 1, 2)
        else:
            coords = np.asarray(list(lines.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)

        return mask