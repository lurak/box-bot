from DDQN.ddqn_agent_cnn import DDQNAgentCnn
import cv2 as cv
import numpy as np


class DDQNAgentOrb(DDQNAgentCnn):
    def __init__(self,
                 gamma,
                 action_number,
                 minibatch,
                 episodes,
                 begin_train,
                 copy_step,
                 epsilon_delta,
                 epsilon_start,
                 epsilon_end,
                 load_model,
                 path_to_load,
                 path_to_save,
                 plots_to_save,
                 episode_steps,
                 episode_to_save,
                 max_buffer_len,
                 model_type
                 ):
        super().__init__(gamma,
                         action_number,
                         minibatch,
                         episodes,
                         begin_train,
                         copy_step,
                         epsilon_delta,
                         epsilon_start,
                         epsilon_end,
                         load_model,
                         path_to_load,
                         path_to_save,
                         plots_to_save,
                         episode_steps,
                         episode_to_save,
                         max_buffer_len,
                         model_type)
        self.orb = cv.BRISK_create()

    def preprocess_observation(self, obs):
        img = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        kp = self.orb.detect(img[25:188, 23:136], None)
        i = 0
        orb_tensor = np.zeros(150)
        for k in kp:
            points = k.pt
            orb_tensor[i] = points[0] / 163
            orb_tensor[i + 1] = points[1] / 113
            orb_tensor[i + 2] = img[int(points[0]), int(points[1])] / 255

            i += 3
            if i == 150:
                break
        return orb_tensor