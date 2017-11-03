
# to ram?
from AirSimClient import *
import matplotlib.image as mpimg
import numpy as np
import math
class drone_env():
    # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
    # https://gym.openai.com/docs/
    def __init__(self, tolerance, step_size, step_velocity, client):
        self.client = client
		# tolerance: the radius of the destination circle
        self.tolerance = tolerance
		# need some kind of reading destination center
        self.des_x = 0
        self.des_y = 0
        self.des_z = 0

        self.crt_x = 0
        self.crt_y = 0
        self.crt_z = 0
        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val

        self.crt_angle = 0

        self.image_index = 0
        self.step_size = step_size
        self.step_velocity = step_velocity

        #self.step(9)

    def step(self, a):
        """ Input:
            Actions: np.array
            Output:
            Observation: np.array (depth image, rgb image or other useful image);
            Reward: float (every move takes -1 reward, colipse takes -20 or other,
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
        #  8: Land
        #  9: Takeoff}

        dict = { 0: [self.crt_x, self.crt_y, self.crt_z - self.step_size, self.step_velocity],
                 1: [self.crt_x, self.crt_y, self.crt_z + self.step_size, self.step_velocity],
                 2: [self.crt_x, self.crt_y - self.step_size, self.crt_z, self.step_velocity],
                 3: [self.crt_x, self.crt_y + self.step_size, self.crt_z, self.step_velocity],
                 4: [self.crt_x + self.step_size, self.crt_y, self.crt_z, self.step_velocity],
                 5: [self.crt_x - self.step_size, self.crt_y, self.crt_z, self.step_velocity],
                 }

        if a >= 0 and a <= 5:
            self.client.moveToPosition(dict[a][0], dict[a][1], dict[a][2], dict[a][3])
        elif a == 6:
            self.crt_angle -= math.pi / 6.0
            self.client.moveByAngle(0, 0, self.crt_z, self.crt_angle, 1) ## only given pitch angle
        elif a == 7:
            self.crt_angle += math.pi / 6.0
            self.client.moveByAngle(0, 0, self.crt_z, self.crt_angle, 1)
        elif a == 8:
            self.client.land()
        else:
            self.client.takeoff()

        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val

        boolean_collision = self.whether_collition()

        boolean_arrive = False
        if (self.crt_x - self.des_x) ** 2.0 \
            + (self.crt_y - self.des_y) ** 2.0 \
                + (self.crt_z - self.des_z) ** 2.0 <= self.tolerance ** 2.0:
                boolean_arrive = True

        images_got = self.get_image()
        # self.step_size = step_size
        # self.step_velocity = step_velocity

        return [images_got, self.crt_x, self.crt_y, self.crt_z, boolean_collision, boolean_arrive]


    def get_image(self):
        # get drone's camera view, return type is response
        responses = self.client.simGetImages([
            ImageRequest(0, AirSimImageType.DepthVis),  #depth visualiztion image
            ImageRequest(1, AirSimImageType.Scene) #scene vision image in png format
            ])

        images = []
        for idx, response in enumerate(responses):
            filename = '/home/zeyuan/AirSim/HelloDrone/pic_taken_by_camera/' + str(self.image_index)+ str(idx)

            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                AirSimClientBase.write_pfm(os.path.normpath(filename + '.pfm'), AirSimClientBase.getPfmArray(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                AirSimClientBase.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            # else: #uncompressed array
            #     print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            #     img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
            #     img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
            #     img_rgba = np.flipud(img_rgba) #original image is fliped vertically
            #     img_rgba[:,:,1:2] = 100 #just for fun add little bit of green in all pixels
            #     AirSimClientBase.write_png(os.path.normpath(filename + '.greener.png'), img_rgba) #write to png
            images.append(mpimg.imread(filename + ".png"))

        self.image_index += 1
        return images

    def get_position(self):
        # get current position of drone
        return self.client.getPosition()

    def whether_collition(self):
        return self.client.getCollisionInfo().has_collided

    def reset(self, tolerance, step_size, step_velocity):
        self.client.reset()
        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val
        self.image_index = 0

        self.step_size = step_size
        self.step_velocity = step_velocity
        self.tolerance = tolerance
        self.step(9)

    def reset(self):
        self.client.reset()
        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val
        self.image_index = 0
        self.step(9)
