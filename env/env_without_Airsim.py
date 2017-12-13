
import time
import numpy as np
import math


class simplified_drone_env():
    # when new environment initialized, the goal is put on (4.1, 4.3) by default.
    # method place goal can help you randomly place a goal
    def __init__(self, tolerance, step_size):
        self.tolerance = tolerance
        self.step_size = step_size
		# default goal
        self.des_x = 4.1
        self.des_y = 4.3
        self.des_z = 0

        # start point (0, 0, 0)
        self.crt_x = 0
        self.crt_y = 0
        self.crt_z = 0

        self.crt_angle = 0

    def step(self, a):
        """ Input:
            Actions: np.array
            Output:
            Reward: float (every move takes 0 reward, colipse takes -20 or other,
            take off at the required area(not exact point, give a range), 100 or other
            future work, search for object during the navigation, 20 or other)
            if_done: boolean if finished
            info: Environment info for debug, not square"""
        # return type: [[images], x, y, z, collision, arrive]

        # dict = {0: Up
        #  1: Down
        #  2: Left
        #  3: Right
        #  4: Forward
        #  5: Backward
        #  6: Left_Rotate
        #  7: Right_Rotate


        dict = { 0: [                    0,                    0,  - self.step_size],
                 1: [                    0,                    0,  + self.step_size],
                 2: [ - self.step_size * np.cos(self.crt_angle + math.pi / 2), - self.step_size * np.sin(self.crt_angle + math.pi / 2),                     0],
                 3: [ + self.step_size * np.cos(self.crt_angle + math.pi / 2), + self.step_size * np.sin(self.crt_angle + math.pi / 2),                     0],
                 4: [ + self.step_size * np.cos(self.crt_angle), + self.step_size * np.sin(self.crt_angle),                     0],
                 5: [ - self.step_size * np.cos(self.crt_angle), - self.step_size * np.sin(self.crt_angle),                     0],
                 }
        boolean_collision = False

        if a >= 0 and a <= 5:
            prev_x = self.crt_x
            prev_y = self.crt_y
            prev_z = self.crt_z

            self.crt_x += dict[a][0]
            self.crt_y += dict[a][1]
            self.crt_z += dict[a][2]

            if self.crt_x > 5.5:
                self.crt_x = 5.5
                boolean_collision = True
            elif self.crt_x < -1.5:
                self.crt_x = -1.5
                boolean_collision = True

            if self.crt_y > 5.5:
                self.crt_y = 5.5
                boolean_collision = True
            elif self.crt_y < -1.5:
                self.crt_y = -1.5
                boolean_collision = True

            if self.crt_z > 0.2:
                self.crt_z = 0.2
                boolean_collision = True
            elif self.crt_z < -3.8:
                self.crt_z = -3.8
                boolean_collision = True


            if self.inTheCircle(prev_x, prev_y, self.crt_x, self.crt_y):
                boolean_collision = True


        elif a == 6:
            self.crt_angle -= math.pi / 6.0

        elif a == 7:
            self.crt_angle += math.pi / 6.0


        #determine whether arrive
        boolean_pos_arrive = False
        if self.getDistance(self.crt_x, self.crt_y, self.des_x, self.des_y) <= self.tolerance:
                boolean_pos_arrive = True

        boolean_ang_arrive = False
        des_angle = np.arctan2(self.des_y - self.crt_y, self.des_x - self.crt_x)
        if np.abs(des_angle - self.crt_angle) < np.pi / 6.0:
                boolean_ang_arrive = True

        boolean_arrive = boolean_pos_arrive and boolean_ang_arrive
        if boolean_arrive:
            print("current loc:",self.crt_x,self.crt_y,self.crt_z)


        distance_to_goal = self.getDistance(self.crt_x, self.crt_y, self.des_x, self.des_y)

        reward = -2.0 / (4 * 2.0 ** 0.5) * distance_to_goal - 2.0 / np.pi * np.abs(des_angle - self.crt_angle)

        if boolean_arrive:
            reward = 100
        elif boolean_collision:
            reward = -100

        print("position:",self.crt_x, self.crt_y, self.crt_z,
            "; distance to certer:", self.getDistance(self.crt_x, self.crt_y, 2, 2),"; Step Reward:",reward)

        return [self.crt_x, self.crt_y, self.crt_z, self.crt_angle, boolean_collision, boolean_arrive, reward]

    def inTheCircle(self, x1, y1, x2, y2):
        l1 = self.getDistance(x1, y1, 2, 2)
        l2 = self.getDistance(x2, y2, 2, 2)
        l12 = self.getDistance(x1, y1, x2, y2)

        if l1 < 1 or l2 < 1:
            return True

        if l12 > 0:
            S = (l1 + l2 + l12) / 2.0
            area = ( S * (S - l1) * (S - l2) * (S - l12) ) ** 0.5
            h = 2.0 * area / l12

            if h >= 1:
                return False

            if l1 ** 2.0 + l12 ** 2.0 - l2 ** 2.0 > 0 and l2 ** 2.0 + l12 ** 2.0 - l1 ** 2.0 > 0:
                return True

        return False


    def getDistance(self, p1_x, p1_y, p2_x, p2_y):
        return ( (p1_x - p2_x) ** 2.0 + (p1_y - p2_y) ** 2.0 ) ** 0.5

    def reset(self):
        # drone return to (0,0,0), goal doesn't move
        self.crt_x = 0
        self.crt_y = 0
        self.crt_z = 0
        self.crt_angle = 0

    def reset(self, tolerance, step_size):
        self.crt_x = 0
        self.crt_y = 0
        self.crt_z = 0
        self.crt_angle = 0

        self.step_size = step_size
        self.tolerance = tolerance

    def place_goal(self):

        # please make sure reset first, otherwise collision may happen when reset
        self.des_x = np.random.random_sample() * 7.0 - 1.5
        self.des_y = np.random.random_sample() * 7.0 - 1.5
        self.des_z = 0

        while (self.crt_x - self.des_x) ** 2.0 + (self.crt_y - self.des_y) ** 2.0 + (self.crt_z - self.des_z) ** 2.0 < 4:
            self.des_x = np.random.random_sample() * 7.0 - 1.5
            self.des_y = np.random.random_sample() * 7.0 - 1.5

        print("new goal placed: x,y,z =",self.des_x,self.des_y,self.des_z)
