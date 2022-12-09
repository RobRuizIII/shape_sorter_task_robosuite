import numpy as np
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from environments.manipulation.two_arm_shape_sorter import TwoArmShapeSorter
from robosuite.wrappers.gym_wrapper import GymWrapper




if __name__ == "__main__":

    # Help message to user
    print()
    print('Press "H" to show the viewer control panel.')

    controller_dict = {}


    # initialize the task
    robosuite_env = TwoArmShapeSorter(
        robots=["Panda", "Panda"],
        env_configuration="single-arm-opposed",
        has_renderer=True,
        has_offscreen_renderer=False,
        reward_shaping=True,
        target_shape="diamond",
        render_collision_mesh=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env = GymWrapper(robosuite_env)

    env.reset()
    env.viewer.set_camera(camera_id=0)

    

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)

    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000)


    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=10)



    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            obs = env.reset()

    

    # do visualization
    
    # Get action limits
    # low, high = env.action_spec

    # for i in range(10000):
    #     action = np.random.uniform(low, high)
    #     obs, reward, done, _ = env.step(action)
    #     print(reward)
    #     env.render()