# This file exports PrisonerEscape environment by registering in OpenAI Gym
# so that we could do `gym.make('PrisonerEscape-v0')`


from gym.envs import register
from Prison_Escape.environment.prisoner_env import PrisonerBothEnv, RewardScheme
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerBlueEnv
from Prison_Escape.environment.observation_spaces import ObservationNames
from Prison_Escape.environment.terrain import Terrain

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return

    print("Registering custom gym environments, could do `gym.make('PrisonerEscape-v0')`")
    register(id='PrisonerEscape-v0', entry_point='simulator.prisoner_env:PrisonerEnv', kwargs={})

    _REGISTERED = True


register_custom_envs()