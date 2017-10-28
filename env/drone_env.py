
# to ram?

class env():
        # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
        # https://gym.openai.com/docs/
        def __init__(self, ):

        def _step(self, a):
            """ Input:
                Actions: np.array
                Output:
                Observation: np.array (depth image, rgb image or other useful image);
                Reward: float (every move takes -1 reward, colipse takes -20 or other,
                take off at the required area(not exact point, give a range), 100 or other
                future work, search for object during the navigation, 20 or other)
                if_done: boolean if finished
                info: Environment info for debug, not square"""

        def _get_image(self, if_depth):
            """
            Be called in step
            Input, if_depth: boolean, determine if depth image taken
            Output, image taken from the api
            """

        def _reset(self):
            """
            Reset the drone position to the fixed point
            Replace the target object
            """

        def _render(self, mode, close):
            # Possilbly not useful
            # Way to record the image or video may be directly in the api

        # Keys to actions, to explain the meaning for actions

        # Clone and restore state, check the reference file at the beginning
