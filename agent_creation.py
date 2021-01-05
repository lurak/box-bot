import gym
import json
from DDQN.ddqn_agent_cnn import DDQNAgentCnn
from DDQN.ddqn_agent_linear import DDQNAgentLinear
from DDQN.ddqn_agent_orb import DDQNAgentOrb
from REINFORCE.reinforce_agent import ReinforceAgent
from neural_networks.cnn_ddqn_model import CnnDDQNModel
from neural_networks.linear_ddqn_model import LinearDDQNModel
from neural_networks.orb_ddqn_model import OrbDDQNModel
from neural_networks.cnn_reinforce_model import CnnReinforceModel


def create_env(name):
    return gym.make(name)


def create_agent(model_type_name, params):
    if model_type_name == "cnn_ddqn":
        return DDQNAgentCnn(model_type=CnnDDQNModel, **params)
    elif model_type_name == "linear_ddqn":
        return DDQNAgentLinear(model_type=LinearDDQNModel, **params)
    elif model_type_name == "orb_ddqn":
        return DDQNAgentOrb(model_type=OrbDDQNModel, **params)
    elif model_type_name == "cnn_rf":
        return ReinforceAgent(model_type=CnnReinforceModel, **params)


def generate_agent(env_name, model_type_name, config_file_path):
    env = create_env(env_name)
    with open(config_file_path, "r") as f:
        params = json.load(f)

    agent = create_agent(model_type_name, params)
    return agent, env


if __name__ == "__main__":
    agent, env = generate_agent("CartPole-v0", "linear_ddqn", "config_files/ddqn_cartpole.json")
    agent.train(env)
