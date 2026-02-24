import numpy as np
import matplotlib.pyplot as plt

from tof_modeling import ToFCamera, Ray
from geometry import Triangle, Point, Sphere, Figure


def simple_pyramid(camera: ToFCamera) -> None:
    triangle_1 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([1, 1, 3])),
        Point(np.array([1, -1, 3]))
    )

    triangle_2 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-1, -1, 3])),
        Point(np.array([-1, 1, 3]))
    )

    triangle_3 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([1, 1, 3])),
        Point(np.array([-1, 1, 3]))
    )

    triangle_4 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-1, -1, 3])),
        Point(np.array([1, -1, 3]))
    )

    figure = Figure([triangle_1, triangle_2, triangle_3, triangle_4])

    camera.visualize_depth_map(figure)
    camera.visualize_point_cloud(figure)

def simple_sphere(camera: ToFCamera, sphere: Sphere) -> None:
    camera.visualize_depth_map(sphere)
    camera.visualize_point_cloud(sphere)

def simple_sphere(camera: ToFCamera, triangle: Triangle) -> None:
    camera.visualize_depth_map(triangle)
    camera.visualize_point_cloud(triangle)


if __name__ == "__main__":
    tof_camera = ToFCamera(
        position=Point(np.array([0, 0, 0])),
        width=100,
        height=100,
        direction=np.array([0, 0, 1]),
        fov=60
    )

    simple_pyramid(tof_camera)