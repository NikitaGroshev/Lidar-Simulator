import numpy as np
import matplotlib.pyplot as plt
from geometry import Point, Sphere, distance_to


class Ray:
    """
    Class for modeling rays of ToF camera. 
    """

    def __init__(self, direction: np.ndarray, start: Point) -> None:
        self.direction = direction
        self.start = start

    def sphere_intersect(self, sphere: Sphere) -> Point | None:
        """
        Method for finding the nearest point of 
        intersection with a sphere.

        Args: 
            sphere: sphere for intersection with ray.

        Returns: the nearest point of intersection in 3D.
        """
        delta_x = self.start.coords - sphere.center.coords
        A = np.dot(self.direction, self.direction)
        B = 2 * np.dot(self.direction, delta_x)
        C = np.dot(delta_x, delta_x) - (sphere.R**2)
        D = B**2 - 4 * A * C
        if (D < 0):
            return None
        else:
            t1 = (-B + np.sqrt(D)) / (2 * A)
            t2 = (-B - np.sqrt(D)) / (2 * A)

            roots = []

            if t1 >= 0:
                roots.append(t1)
            if t2 >= 0:
                roots.append(t2)

            if not roots:
                return None
            t_min = min(roots)

            return Point(self.start.coords + t_min * self.direction)


class ToFCamera:
    """
    Class for modeling ToF camera.
    """

    def __init__(
        self,
        position: Point,
        width: int,
        height: int,
        direction: np.ndarray,
        fov: float        # in degrees
    ) -> None:
        self.fov = np.deg2rad(fov)
        self.position = position
        self.direction = direction
        self.width = width
        self.height = height

    def generate_rays(self) -> list[Ray]:
        """
        Method for generating rays of ToF camera.

        Returns:
            list of rays.
        """
        rays = []
        d = np.linalg.norm(self.direction)
        e1 = self.direction / d

        a = np.array([1, 0, 0])
        if (np.all((a - e1) == 0)):
            a = np.array([0, 1, 0])

        e2 = np.cross(e1, a)
        e2 /= np.linalg.norm(e2)

        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)
        
        center = self.position.coords + e1

        ratio = self.height / self.width
        half_width = np.tan(self.fov / 2)
        half_height = ratio * half_width

        abscissa = np.linspace(-half_width, half_width, self.width)
        ordinate = np.linspace(-half_height, half_height, self.height)

        for y in ordinate:
            for x in abscissa:
                pixel_point = center + x * e2 + y * e3

                ray_direction = pixel_point - self.position.coords
                ray_direction /= np.linalg.norm(ray_direction)

                ray = Ray(
                    direction=ray_direction,
                    start=self.position
                )

                rays.append(ray)

        return rays
    
    def get_points_and_distances_to_sphere(self, sphere: Sphere) -> tuple[np.ndarray, np.ndarray]:
        """
        Method for getting points of the sphere and distances to the sphere.

        Args:
            sphere: Sphere for which distances are calculated.

        Returns: Array of points of the sphere and array of distances to the sphere.
        """
        distances = []
        points = []
        for ray in self.generate_rays():
            point = ray.sphere_intersect(sphere)
            if point is None:
                distances.append(np.nan)
            else:
                dist = distance_to(point, self.position)
                distances.append(dist)
                points.append(point.coords)

        result_distances = np.array(distances)
        result_points = np.array(points) if points else np.array([])

        return result_points, result_distances
    
    def get_time(self, sphere: Sphere) -> np.ndarray:
        """
        Method for getting time spent by light.

        Args:
            sphere: sphere for which times are calculated.

        Returns: Array of times spent by light.
        """
        c = 299792458   # speed of the light
        _, distances = self.get_points_and_distances_to_sphere(sphere)
        times = 2 * distances / c
        return times

    def visualize_depth_map(self, sphere: Sphere) -> None:
        """
        Method for visualizing depth map.

        Args:
            sphere: sphere in 3D for ToF camera.
        """

        _, depth_map = self.get_points_and_distances_to_sphere(sphere)
        depth_map = depth_map.reshape((self.width, self.height))

        figure, axis = plt.subplots(figsize=(8, 6))
        axis: plt.Axes
        
        if np.all(np.isnan(depth_map)):
            masked_depth = np.ma.masked_where(np.ones_like(depth_map, dtype=bool), depth_map)
            map = axis.imshow(masked_depth, cmap=plt.cm.inferno)
        else:
            cmap = plt.cm.inferno.copy()
            cmap.set_bad(color="none")
            map = axis.imshow(depth_map, cmap=cmap, vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))

        axis.set_title('ToF camera depth map')
        axis.axis("image")
        axis.grid(False)
        figure.colorbar(map, ax=axis, label='Distance to camera')

        plt.show()

    def visualize_point_cloud(self, sphere: Sphere) -> None:
        """
        Method for visualizing point cloud.

        Args:
            sphere: sphere in 3D for ToF camera.
        """

        points, _ = self.get_points_and_distances_to_sphere(sphere)

        figure = plt.figure(figsize=(8, 6))
        axis: plt.Axes = figure.add_subplot(projection='3d')

        if points.size == 0:
            axis.scatter3D(
                self.position.coords[0],
                self.position.coords[1],
                self.position.coords[2],
                color='red', marker='^', s=100, label='Camera position'
            )
            plt.legend()
        else:
            axis.scatter3D(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c='royalblue'
            )

        axis.set_title("ToF Point Cloud")
        axis.set_xlabel("X", fontsize=10)
        axis.set_ylabel("Y", fontsize=10)
        axis.set_zlabel("Z", fontsize=10)

        plt.show()


if __name__ == "__main__":
    tof_camera = ToFCamera(
        position=Point(np.array([0, 0, 0])),
        width=100,
        height=100,
        direction=np.array([0, 0, 1]),
        fov=60
    )

    sphere = Sphere(
        R=3,
        center=Point(np.array([0, 0, 6]))
    )

    tof_camera.visualize_depth_map(sphere)
    tof_camera.visualize_point_cloud(sphere)