import ezdxf
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BendLine:
    start_point: np.ndarray
    end_point: np.ndarray
    angle: float = 0.0  # Will be set manually
    direction: str = "up"  # Will be set manually

@dataclass
class Section:
    points: List[np.ndarray]
    index: int
    insertion_order: int = -1  # -1 indicates not set
    connected_bends: Set[int] = None

    def __post_init__(self):
        if self.connected_bends is None:
            self.connected_bends = set()

class Part:
    def __init__(self, dxf_path: str, material_thickness: float):
        self.dxf_path = dxf_path
        self.material_thickness = material_thickness
        self.points: List[np.ndarray] = []
        self.bends: List[BendLine] = []
        self.sections: List[Section] = []
        self.bend_to_sections: Dict[int, Tuple[int, int]] = {}
        self.section_to_bends: Dict[int, Set[int]] = defaultdict(set)
        
        # Parse DXF and initialize structures
        self.parse_dxf()
        self.identify_sections()

    def parse_dxf(self):
        """Parse DXF file to extract part outline and bend lines"""
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()
        
        # Extract part outline from 'cut' layer
        outline_points = []
        for entity in msp.query('LWPOLYLINE[layer=="cut"]'):
            points = list(entity.get_points())
            outline_points.extend([np.array(p) for p in points])
        
        # Extract bend lines from 'bend' layers
        bend_lines = []
        for entity in msp.query('LINE[layer=="bend"]'):
            start = np.array(entity.dxf.start[:2])
            end = np.array(entity.dxf.end[:2])
            bend_lines.append(BendLine(start, end))
        
        # Add intersection points to outline
        self.points = self._add_intersection_points(outline_points, bend_lines)
        self.bends = bend_lines

    def _add_intersection_points(self, outline_points: List[np.ndarray], 
                               bend_lines: List[BendLine]) -> List[np.ndarray]:
        """Add intersection points between bend lines and outline to points list"""
        result_points = outline_points.copy()
        
        for bend in bend_lines:
            for i in range(len(outline_points)):
                p1 = outline_points[i]
                p2 = outline_points[(i + 1) % len(outline_points)]
                
                intersection = self._line_intersection(
                    p1, p2, bend.start_point, bend.end_point
                )
                
                if intersection is not None:
                    result_points.append(intersection)
        
        # Sort points to maintain outline order
        return self._sort_points_by_connectivity(result_points)

    def _line_intersection(self, p1: np.ndarray, p2: np.ndarray, 
                          p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return np.array([x, y])
        return None

    def _sort_points_by_connectivity(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Sort points to maintain continuous outline"""
        sorted_points = [points[0]]
        remaining_points = points[1:]
        
        while remaining_points:
            current = sorted_points[-1]
            next_point = min(remaining_points, 
                           key=lambda p: np.linalg.norm(p - current))
            sorted_points.append(next_point)
            remaining_points.remove(next_point)
            
        return sorted_points

    def identify_sections(self):
        """Use graph traversal to identify sections divided by bend lines"""
        # Create graph representation
        graph = self._create_section_graph()
        
        # Use DFS to identify connected components (sections)
        visited = set()
        section_index = 0
        
        for point in self.points:
            if tuple(point) not in visited:
                section_points = self._dfs_section(point, graph, visited)
                self.sections.append(Section(
                    points=section_points,
                    index=section_index
                ))
                section_index += 1

        # Identify connections between sections and bends
        self._identify_section_connections()

    def _create_section_graph(self) -> Dict[Tuple[float, float], Set[Tuple[float, float]]]:
        """Create graph representation of part for section identification"""
        graph = defaultdict(set)
        
        # Add connections from outline
        for i in range(len(self.points)):
            p1 = tuple(self.points[i])
            p2 = tuple(self.points[(i + 1) % len(self.points)])
            
            # Check if connection crosses any bend line
            if not any(self._intersects_bend(self.points[i], 
                                           self.points[(i + 1) % len(self.points)], 
                                           bend) 
                      for bend in self.bends):
                graph[p1].add(p2)
                graph[p2].add(p1)
                
        return graph

    def _intersects_bend(self, p1: np.ndarray, p2: np.ndarray, 
                        bend: BendLine) -> bool:
        """Check if line segment intersects with bend line"""
        return self._line_intersection(p1, p2, bend.start_point, 
                                     bend.end_point) is not None

    def _dfs_section(self, start: np.ndarray, graph: Dict, 
                    visited: Set) -> List[np.ndarray]:
        """DFS to find points in a section"""
        stack = [start]
        section_points = []
        
        while stack:
            point = stack.pop()
            point_tuple = tuple(point)
            
            if point_tuple not in visited:
                visited.add(point_tuple)
                section_points.append(point)
                
                for neighbor in graph[point_tuple]:
                    if neighbor not in visited:
                        stack.append(np.array(neighbor))
                        
        return section_points

    def _identify_section_connections(self):
        """Identify which sections are connected by which bends"""
        for bend_idx, bend in enumerate(self.bends):
            connected_sections = []
            
            for section in self.sections:
                # Check if any point in section is on the bend line
                for point in section.points:
                    if self._point_on_line(point, bend.start_point, bend.end_point):
                        connected_sections.append(section.index)
                        section.connected_bends.add(bend_idx)
                        
            if len(connected_sections) == 2:
                self.bend_to_sections[bend_idx] = tuple(connected_sections)
                for section_idx in connected_sections:
                    self.section_to_bends[section_idx].add(bend_idx)

    def _point_on_line(self, point: np.ndarray, line_start: np.ndarray, 
                      line_end: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if point lies on line segment"""
        d = np.linalg.norm(line_end - line_start)
        if d == 0:
            return np.allclose(point, line_start, atol=tolerance)
            
        t = np.dot(point - line_start, line_end - line_start) / (d * d)
        if t < 0 or t > 1:
            return False
            
        projection = line_start + t * (line_end - line_start)
        return np.allclose(point, projection, atol=tolerance)

    # Setter methods for manual inputs
    def set_bend_angles(self, angles: List[float]):
        """Set bend angles manually"""
        if len(angles) != len(self.bends):
            raise ValueError(f"Expected {len(self.bends)} angles, got {len(angles)}")
        for bend, angle in zip(self.bends, angles):
            bend.angle = angle

    def set_bend_directions(self, directions: List[str]):
        """Set bend directions manually"""
        if len(directions) != len(self.bends):
            raise ValueError(f"Expected {len(self.bends)} directions, got {len(directions)}")
        for bend, direction in zip(self.bends, directions):
            if direction not in ["up", "down"]:
                raise ValueError(f"Invalid direction {direction}. Use 'up' or 'down'")
            bend.direction = direction

    def set_insertion_order(self, order: List[int]):
        """Set section insertion order manually"""
        if len(order) != len(self.sections):
            raise ValueError(f"Expected {len(self.sections)} orders, got {len(order)}")
        for section, order_num in zip(self.sections, order):
            section.insertion_order = order_num
