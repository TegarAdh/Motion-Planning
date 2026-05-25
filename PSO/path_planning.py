# path_planning.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Obstacle:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = radius


class Environment:
    def __init__(self, width, height, robot_radius, start, goal):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


class EnvCostFunction:
    def __init__(self, env, num_control_points, resolution):
        self.env = env
        self.num_control_points = num_control_points
        self.resolution = resolution

    def __call__(self, x):
        pts = self._decode_solution(x)
        path = self._generate_path(pts)

        length = self._path_length(path)
        penalty = self._collision_penalty(path)

        cost = length + penalty

        details = {
            "sol": path,
            "length": length,
            "penalty": penalty,
        }

        return cost, details

    def _decode_solution(self, x):
        x = np.array(x)

        points = [self.env.start]

        for i in range(self.num_control_points):
            px = x[2 * i] * self.env.width
            py = x[2 * i + 1] * self.env.height
            points.append(np.array([px, py]))

        points.append(self.env.goal)

        return np.array(points)

    def _generate_path(self, points):
        path = []

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            for t in np.linspace(0, 1, self.resolution):
                p = p1 + t * (p2 - p1)
                path.append(p)

        return np.array(path)

    def _path_length(self, path):
        dist = 0.0

        for i in range(len(path) - 1):
            dist += np.linalg.norm(path[i + 1] - path[i])

        return dist

    def _collision_penalty(self, path):
        penalty = 0

        for p in path:
            x, y = p

            # keluar map
            if x < 0 or x > self.env.width or y < 0 or y > self.env.height:
                penalty += 1000

            for obs in self.env.obstacles:
                d = np.linalg.norm(p - obs.center)

                if d <= (obs.radius + self.env.robot_radius):
                    penalty += 1000

        return penalty


def plot_environment(env):
    ax = plt.gca()

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    # start
    plt.plot(env.start[0], env.start[1], 'go', markersize=10)

    # goal
    plt.plot(env.goal[0], env.goal[1], 'ro', markersize=10)

    # obstacle
    for obs in env.obstacles:
        c = Circle(obs.center, obs.radius, color='black')
        ax.add_patch(c)


def plot_path(path, color='b'):
    line, = plt.plot(path[:, 0], path[:, 1], color=color, linewidth=2)
    return line


def update_path(path, line):
    line.set_xdata(path[:, 0])
    line.set_ydata(path[:, 1])
    plt.pause(0.01)