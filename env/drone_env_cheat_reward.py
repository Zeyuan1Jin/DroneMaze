
# to ram?
import time
from AirSimClient import *
import matplotlib.image as mpimg
import numpy as np
import math
from set_foreground import *
from pymouse import PyMouse
from pykeyboard import PyKeyboard

class drone_env():
    # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
    # https://gym.openai.com/docs/
    def __init__(self, tolerance, step_size, step_velocity, client):
        self.duration = step_size / step_velocity + 1
        self.client = client
		# tolerance: the radius of the destination circle
        self.tolerance = tolerance
		# need some kind of reading destination center
        self.des_x = 4.1
        self.des_y = 4.3
        self.des_z = 0
        print("Initial Destination:",self.des_x,self.des_y,self.des_z)
        self.place_goal()
        
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

    def keep_z(self):
        self.client.moveByVelocity(0, 0, self.step_velocity, (self.des_z - self.crt_z) / self.step_velocity)
        time.sleep(self.duration)
        
##        a = self.get_position()
##        self.crt_x = a.x_val
##        self.crt_y = a.y_val
##        self.crt_z = a.z_val
##        b = self.get_orientation()
##        self.crt_angle = b[2]
##        
##
##        boolean_collision = self.whether_collition()
##
##        boolean_arrive = False
##        if (self.crt_x - self.des_x) ** 2.0 \
##            + (self.crt_y - self.des_y) ** 2.0 \
##                + (self.crt_z - self.des_z) ** 2.0 <= self.tolerance ** 2.0:
##                boolean_arrive = True
##                print("current loc:",self.crt_x,self.crt_y,self.crt_z)
##
##        images_got = self.get_image()
##        # self.step_size = step_size
##        # self.step_velocity = step_velocity
##
##        reward = 0
##        if boolean_arrive:
##            reward = 10
##       
##        ##if boolean_collision:
##        ##    reward = -100
##        # print("Step Reward:",reward)
##        return [images_got, boolean_arrive, reward]
    
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
        
        p = self.get_position()
        prv_x = p.x_val
        prv_y = p.y_val
        prv_z = p.z_val
        
        dict = { 0: [                    0,                    0,  - self.step_velocity, self.step_size],
                 1: [                    0,                    0,  + self.step_velocity, self.step_size],
                 2: [ - self.step_velocity * np.cos(self.crt_angle + math.pi / 2), - self.step_velocity * np.sin(self.crt_angle + math.pi / 2),                     0, self.step_size],
                 3: [ + self.step_velocity * np.cos(self.crt_angle + math.pi / 2), self.step_velocity * np.sin(self.crt_angle + math.pi / 2),                     0, self.step_size],
                 4: [ + self.step_velocity * np.cos(self.crt_angle), self.step_velocity * np.sin(self.crt_angle),                     0, self.step_size],
                 5: [ - self.step_velocity * np.cos(self.crt_angle), - self.step_velocity * np.sin(self.crt_angle),                     0, self.step_size],
                 }

        if a >= 0 and a <= 5:
            self.client.moveByVelocity(dict[a][0], dict[a][1], dict[a][2], dict[a][3] / self.step_velocity)
            time.sleep(self.duration)
        elif a == 6:
            self.crt_angle -= math.pi / 6.0
            self.client.moveByAngle(0, 0, self.crt_z, self.crt_angle, 1) ## only given pitch angle
            time.sleep(self.duration)
            
        elif a == 7:
            self.crt_angle += math.pi / 6.0
            self.client.moveByAngle(0, 0, self.crt_z, self.crt_angle, 1)
            time.sleep(self.duration)
            
        elif a == 8:
            self.client.land()
        else:
            self.client.takeoff()

           
        action = a
        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val
        self.keep_z() 
        b = self.get_orientation()
        self.crt_angle = b[2]
        

        boolean_collision = self.whether_collition()

        #determine whether arrive
        boolean_pos_arrive = False
        if (self.crt_x - self.des_x) ** 2.0 \
            + (self.crt_y - self.des_y) ** 2.0 \
                + (self.crt_z - self.des_z) ** 2.0 <= self.tolerance ** 2.0:
                boolean_pos_arrive = True
                
        boolean_ang_arrive = False
        des_angle = np.arctan2(self.des_y - self.crt_y, self.des_x - self.crt_x)
        if np.abs(des_angle - self.crt_angle) < np.pi / 6.0:
                boolean_ang_arrive = True

        boolean_arrive = boolean_pos_arrive and boolean_ang_arrive
        if boolean_arrive:
            print("current loc:",self.crt_x,self.crt_y,self.crt_z)
                
        images_got = self.get_image()
        # self.step_size = step_size
        # self.step_velocity = step_velocity

        distance_to_goal =(     (self.crt_x - self.des_x) ** 2.0 
                        + (self.crt_y - self.des_y) ** 2.0 
                   + (self.crt_z - self.des_z) ** 2.0       ) ** 0.5

        reward = -10.0 / (4 * 2.0 ** 0.5) * distance_to_goal - 10 / np.pi * np.abs(des_angle - self.crt_angle)
        
        if boolean_arrive:
            reward = 10
        elif action >= 0 and action <= 5 and (self.crt_x - prv_x) ** 2.0 \
            + (self.crt_y - prv_y) ** 2.0 \
                + (self.crt_z - prv_z) ** 2.0 <= (self.step_size/2) ** 2.0:
            reward = -100
        ##if boolean_collision:
        ##    reward = -100
        print("Step Reward:",reward)
        
        return [images_got, self.crt_x, self.crt_y, self.crt_z, self.crt_angle, boolean_collision, boolean_arrive, reward]


    def get_image(self):
        # get drone's camera view, return type is response
        responses = self.client.simGetImages([
            ImageRequest(0, AirSimImageType.DepthVis),  #depth visualiztion image
            ImageRequest(1, AirSimImageType.Scene) #scene vision image in png format
            ])

        images = []
        for idx, response in enumerate(responses):
            filename = 'c:/pic_taken_by_camera/' + str(self.image_index)+ str(idx)

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

        
        return images

    def get_position(self):
        # get current position of drone
        return self.client.getPosition()
    
    def get_orientation(self):
        return self.client.toEulerianAngle(self.client.getOrientation())
    
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
    
    def place_goal(self):
        w = WindowMgr()
        w.find_window_wildcard(".*Blocks Environment for AirSim Blocks Environment for AirSim*")
        #w.find_window_wildcard(".*Unreal Engine*")
        m = PyMouse()
        k = PyKeyboard()
        w.set_foreground()
        x_dim, y_dim = m.screen_size()
        #m.click(x_dim/2, y_dim/2, 1)
        k.type_string('q')
        print("place view point")

        time.sleep(1)
        file = open(r'''C:\Users\zeyua\Source\Repos\AirSim\Unreal\Environments\Blocks\Saved\StagedBuilds\WindowsNoEditor\Blocks\Spawn_Location.txt''',"r")
        des_string = file.readline()
        file.close()
        array = des_string.replace("="," ").split()
        print(array)
        self.des_x = float(array[1]) / 100 + 2
        self.des_y = float(array[3]) / 100 + 2
        self.des_z = float(array[5]) / (-100) + 0.6
        print("destination: x,y,z =",self.des_x,self.des_y,self.des_z)
        
    def reset(self):
        self.client.reset()
        a = self.get_position()
        self.crt_x = a.x_val
        self.crt_y = a.y_val
        self.crt_z = a.z_val
        self.image_index = 0
        
