import numpy as np


class Point:
    """
    Class for representing point in 3D.
    """

    def __init__(self, coords: np.ndarray) -> None:
        if coords.size == 3:
            self._coords = coords
        else:
            self._coords = np.array([0, 0, 0])

    def __str__(self) -> str:
        return f"[{self._coords[0], self._coords[1], self._coords[2]}]"
    
    @property
    def coords(self) -> np.ndarray:
        return self._coords
    
    @coords.setter
    def coords(self, new_coords: np.ndarray) -> None:
        if new_coords.size == 3:
            self._coords = new_coords
        else:
            raise ValueError("Coords.setter: Incorrect size of new coordinates")
        
    def __eq__(self, other: "Point") -> bool:
        return np.all(self.coords == other.coords)


class Sphere:
    """
    Class for representing sphere in 3D.
    """

    def __init__(self, R: float, center: Point) -> None:
        self.R = R
        self.center = center

def distance_to(p1: Point, p2: Point) -> float:
    """
    Supporting function for calculation 
    Euclidean distance between two points.
    """
    return np.sqrt(np.sum((p1.coords - p2.coords)**2)) 