from DDQN.ddqn_agent_cnn import DDQNAgentCnn


class DDQNAgentLinear(DDQNAgentCnn):
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

    @staticmethod
    def preprocess_observation(obs):
        return obs
