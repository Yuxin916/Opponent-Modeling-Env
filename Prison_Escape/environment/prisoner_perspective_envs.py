from Prison_Escape.environment.prisoner_env import PrisonerBothEnv, ObservationType
from Prison_Escape.blue_policies.heuristic import BlueHeuristic
from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
import numpy as np
import gym

"""
PrisonerBlueEnv and PrisonerEnv are essentially wrappers for the PrisonerBothEnv class such that 
given the policy of the other team, we just return the observations for the desired team (red or blue).
"""

class PrisonerBlueEnv(gym.Wrapper):
    """ This environment return blue observations and takes in blue actions """
    def __init__(self,
                 env: PrisonerBothEnv,
                 fugitive_policy):
        super().__init__(env)
        self.env = env
        self.fugitive_policy = fugitive_policy

        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        self.observation_space = self.env.blue_observation_space
        self.action_space = self.env.blue_action_space
        self.num_agents = self.env.num_search_parties + self.env.num_helicopters

        self.max_timesteps = self.env.max_timesteps

        self.obs_names = self.env.blue_obs_names

    def reset(self, seed=None):
        # print("*"*60)
        # print(f"RESET Blue-Wrapper ENVIRONMENT")
        self.env.reset(seed)
        if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
            self.fugitive_policy.reset()
        return self.env.blue_observation
        
    def step(self, blue_action):
        # get red observation for policy
        red_obs_in = self.env._fugitive_observation
        red_action = self.fugitive_policy.predict(red_obs_in)
        _, blue_obs, reward, done, i = self.env.step_both(red_action[0], blue_action)
        return blue_obs, reward, done, i 