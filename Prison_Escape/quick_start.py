from Prison_Escape.environment.prisoner_env import PrisonerBothEnv
from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerBlueEnv
from Prison_Escape.environment.load_environment import load_environment
from blue_policies.heuristic import BlueHeuristic

env_path = "Prison_Escape/environment/configs/balance_game.yaml"

env = load_environment(env_path)
# env = PrisonerBothEnv()

red_policy = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)
env = PrisonerBlueEnv(env, red_policy)


# what MARL should do. now use Heuristic policy for blue
blue_policy = BlueHeuristic(env, debug=False)

blue_obs = env.reset()
blue_policy.reset()
blue_policy.init_behavior()

done = False
t = 0

while not done:
    t += 1
    blue_actions = blue_policy.predict(blue_obs)
    blue_obs, reward, done, _ = env.step(blue_actions)
    print("blue_obs", blue_obs)
    done = True

print('stop here')
print('stop here')
print('stop here')
