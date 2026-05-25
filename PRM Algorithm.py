import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# parameter
N_SAMPLE = 500
N_KNN = 10
MAX_EDGE_LEN = 30.0
show_animation = True


class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacle_x_list, obstacle_y_list,
                 robot_radius, rng=None):

    obstacle_kd_tree = KDTree(
        np.vstack((obstacle_x_list, obstacle_y_list)).T
    )

    sample_x, sample_y = sample_points(
        start_x, start_y, goal_x, goal_y, robot_radius,
        obstacle_x_list, obstacle_y_list,
        obstacle_kd_tree, rng
    )

    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(
        sample_x, sample_y, robot_radius, obstacle_kd_tree
    )

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y,
        road_map, sample_x, sample_y
    )

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    dx = gx - sx
    dy = gy - sy
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    yaw = math.atan2(dy, dx)

    x, y = sx, sy
    step = rr
    n_step = int(d / step)

    for _ in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True
        x += step * math.cos(yaw)
        y += step * math.sin(yaw)

    dist, _ = obstacle_kd_tree.query([gx, gy])
    return dist <= rr


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    road_map = []
    n_sample = len(sample_x)

    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for i, ix, iy in zip(range(n_sample), sample_x, sample_y):
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)

        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    while True:
        if not open_set:
            print("Cannot find path")
            return [], []

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if show_animation and len(closed_set) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("Goal found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        del open_set[c_id]
        closed_set[c_id] = current

        for n_id in road_map[c_id]:
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)

            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue

            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id] = node
            else:
                open_set[n_id] = node

    rx, ry = [goal_node.x], [goal_node.y]
    parent = goal_node.parent_index

    while parent != -1:
        n = closed_set[parent]
        rx.append(n.x)
        ry.append(n.y)
        parent = n.parent_index

    return rx, ry


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = rng.uniform(min_x, max_x)
        ty = rng.uniform(min_y, max_y)

        dist, _ = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x += [sx, gx]
    sample_y += [sy, gy]

    return sample_x, sample_y


def main():
    print("PRM start!")

    # start & goal
    sx, sy = -5.0, -5.0
    gx, gy = 50.0, 50.0
    robot_size = 2.0

    ox, oy = [], []

    # ===== MAP -10 sampai 60 =====
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)

    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)

    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)

    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)

    # obstacle dalam
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)

    for i in range(-10, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    for i in range(0, 21):
        ox.append(i)
        oy.append(30.0)

    for i in range(-10, 11):
        ox.append(i)
        oy.append(10.0)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-10, 60)
        plt.ylim(-10, 60)
        plt.xticks(range(-10, 61, 10))
        plt.yticks(range(-10, 61, 10))

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size)

    if rx:
        plt.plot(rx, ry, "-r")
        plt.show()
    else:
        print("Path not found")


if __name__ == '__main__':
    main()