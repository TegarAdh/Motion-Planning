import math
from sys import maxsize
import matplotlib.pyplot as plt

show_animation = True


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"
        self.h = 0
        self.k = 0

    def cost(self, state):

        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt((self.x - state.x) ** 2 +
                         (self.y - state.y) ** 2)

    def set_state(self, state):

        if state not in ["s", ".", "#", "e", "*"]:
            return

        self.state = state


class Map:

    def __init__(self, row, col, origin_x=0, origin_y=0):

        self.row = row
        self.col = col
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.map = self.init_map()

    def init_map(self):

        grid = []

        for i in range(self.row):

            tmp = []

            for j in range(self.col):
                tmp.append(State(i, j))

            grid.append(tmp)

        return grid

    def world_to_map(self, x, y):

        mx = x - self.origin_x
        my = y - self.origin_y

        return int(mx), int(my)

    def map_to_world(self, x, y):

        wx = x + self.origin_x
        wy = y + self.origin_y

        return wx, wy

    def get_neighbors(self, state):

        neighbors = []

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:

                if i == 0 and j == 0:
                    continue

                nx = state.x + i
                ny = state.y + j

                if nx < 0 or nx >= self.row:
                    continue

                if ny < 0 or ny >= self.col:
                    continue

                neighbors.append(self.map[nx][ny])

        return neighbors

    def set_obstacle(self, points):

        for x, y in points:

            mx, my = self.world_to_map(x, y)

            if mx < 0 or mx >= self.row or my < 0 or my >= self.col:
                continue

            self.map[mx][my].set_state("#")


class Dstar:

    def __init__(self, maps):

        self.map = maps
        self.open_list = set()

    def min_state(self):

        if not self.open_list:
            return None

        return min(self.open_list, key=lambda x: x.k)

    def get_kmin(self):

        if not self.open_list:
            return -1

        return min([x.k for x in self.open_list])

    def insert(self, state, h_new):

        if state.t == "new":
            state.k = h_new

        elif state.t == "open":
            state.k = min(state.k, h_new)

        elif state.t == "close":
            state.k = min(state.h, h_new)

        state.h = h_new
        state.t = "open"

        self.open_list.add(state)

    def remove(self, state):

        if state.t == "open":
            state.t = "close"

        self.open_list.remove(state)

    def process_state(self):

        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()

        self.remove(x)

        if k_old < x.h:

            for y in self.map.get_neighbors(x):

                if y.h <= k_old and x.h > y.h + x.cost(y):

                    x.parent = y
                    x.h = y.h + x.cost(y)

        elif k_old == x.h:

            for y in self.map.get_neighbors(x):

                if (
                        y.t == "new"
                        or (y.parent == x and y.h != x.h + x.cost(y))
                        or (y.parent != x and y.h > x.h + x.cost(y))
                ):

                    y.parent = x
                    self.insert(y, x.h + x.cost(y))

        return self.get_kmin()

    def run(self, start, end):

        self.open_list.add(end)

        while True:

            self.process_state()

            if start.t == "close":
                break

        tmp = start

        pathx = []
        pathy = []

        while tmp != end:

            wx, wy = self.map.map_to_world(tmp.x, tmp.y)

            pathx.append(wx)
            pathy.append(wy)

            if show_animation:

                plt.plot(pathx, pathy, "-r", linewidth=2)
                plt.pause(0.05)

            tmp = tmp.parent

        return pathx, pathy


def main():

    world_min = -10
    world_max = 60

    size = world_max - world_min

    m = Map(size, size, origin_x=world_min, origin_y=world_min)

    ox = []
    oy = []

    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)

    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)

    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)

    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)

    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)

    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)

    for i in range(0, 21):
        ox.append(i)
        oy.append(30)

    for i in range(-10, 11):
        ox.append(i)
        oy.append(10)

    m.set_obstacle(list(zip(ox, oy)))

    start_world = (-5, -5)
    goal_world = (50, 50)

    sx, sy = m.world_to_map(*start_world)
    gx, gy = m.world_to_map(*goal_world)

    start = m.map[sx][sy]
    goal = m.map[gx][gy]

    if show_animation:

        plt.plot(ox, oy, ".k")  # obstacle titik
        plt.plot(start_world[0], start_world[1], "og")
        plt.plot(goal_world[0], goal_world[1], "xb")

        plt.axis("equal")
        plt.grid(True)
        plt.xticks(range(-10, 61, 10))
        plt.yticks(range(-10, 61, 10))

    dstar = Dstar(m)

    pathx, pathy = dstar.run(start, goal)

    if show_animation:

        plt.plot(pathx, pathy, "-r", linewidth=2)
        plt.show()


if __name__ == "__main__":
    main()