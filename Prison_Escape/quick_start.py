# from Prison_Escape.environment.obs_embedding_wrapper import PrisonerGNNEnv
# from Prison_Escape.environment.gnn_wrapper import PrisonerGNNEnv
from Prison_Escape.environment.load_environment import load_environment
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerBlueEnv
from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from Prison_Escape.fugitive_policies.heuristic import HeuristicPolicy
from Prison_Escape.fugitive_policies.a_star_avoid import AStarAdversarialAvoid

from blue_policies.heuristic import BlueHeuristic

# config for raw env
raw_env_path = "Prison_Escape/environment/configs/mytest.yaml"

raw_env = load_environment(raw_env_path)

# red_policy = HeuristicPolicy(raw_env, epsilon=0.0)
red_policy = AStarAdversarialAvoid(raw_env, max_speed=10, cost_coeff=1000, visualize=False)
# red_policy = RRTStarAdversarialAvoid(raw_env,
#                                      n_iter=1500,
#                                      step_len=150,
#                                      search_radius=150,
#                                      max_speed=7.5,
#                                      terrain_cost_coef=500,
#                                      visualize=False,
#                                      gamma=15,
#                                      goal_sample_rate=0.1,
#                                      epsilon=0.1
#                                      )
env = PrisonerBlueEnv(raw_env, red_policy)

DEBUG = False

if DEBUG:
    print(env.action_space)
    print(env.observation_space)
    print(env.num_agents)

    env_instance, info = env
    obs = env_instance.reset()
    # print(obs['search_parties_0'])
    obp = env_instance.observation_space
    # print(obp)
    print(obp.contains(obs['search_parties_0']))
    # pass
    #
    import numpy as np

    for i in range(1000):
        blue_actions = [np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 127])]
        # transform blue_actions to try_action_dict, with prefix added search_parties_ and helicopters_
        try_action_dict = {}
        num_search_parties = 3
        num_helicopters = 1
        try_action_dict = {f'search_parties_{i}': blue_actions[i] for i in range(num_search_parties)}
        try_action_dict.update({f'helicopters_{j}': blue_actions[-j-1] for j in range(num_helicopters)})
        # print(try_action_dict)
        #
        obs, rewards, dones, info = env_instance.step(try_action_dict)
        # print(obs['search_parties_0'])
        print(obp.contains(obs['search_parties_0']))
        env_instance.render('heuristic', show=True, fast=True, show_delta=True)

# env = PrisonerGNNEnv(env)
# print(env.action_space)
# print(env.observation_space)
# print(env.total_agents_num)

# Reset the environment and policy
# gnn_obs, blue_obs = env.reset()
blue_obs = env.reset()

# what MARL should do. now use Heuristic policy for blue
blue_policy = BlueHeuristic(env, debug=False)
blue_policy.reset()
blue_policy.reset()
blue_policy.init_behavior()

for i in range(1000):
    blue_actions = blue_policy.predict(blue_obs)
    # blue_actions = [np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 127])]
    # gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)
    blue_obs, reward, done, _ = env.step(blue_actions)
    # print("blue_obs", blue_obs)
    env.render('heuristic', show=True, fast=True, show_delta=True)

    pass

print('stop here')
print('stop here')
print('stop here')
