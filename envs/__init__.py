from . import parallel, wrappers
import gymnasium as gym
import numpy as np
import mujoco

# -------- Env wrapper ----------
class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        model = self.env.unwrapped.model
        model.actuator_ctrlrange[:, 0] = -1
        model.actuator_ctrlrange[:, 1] =  1
        # self.env.unwrapped.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(self.env.unwrapped.observation_space.shape[0]*2,), dtype=np.float64
        # )
        self.action_space = self.env.unwrapped._set_action_space()
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.x_hist = []
        self.z_hist = []
        self.z_target = 1
        self.z_sum = 0
        self.z_cnt = 0
        spaces = {}
        self.keys = ["qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"]
        self.values = [22, 23, 130, 78, 17, 78]
        # self.keys = ["qpos", "qvel", "cinert", "cvel"]
        # self.values = [22, 23, 130, 78]
        for key, value in zip(self.keys, self.values):
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, (value,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        self.index += 1
        obs, reward, done, terminated, info = self.env.step(action)
        reward += 5 * (1 - (1.4 - min(obs[0], 1.4)) ** 2 / 1.4 ** 2)
        # z_all = self.z_sum/self.z_cnt
        # if obs[0] > z_all and obs[0] < 1.4:
        #     reward += 100
        self.reward += reward
        # x = self.data.qpos[0]
        # dx = x - self.x_hist[0]
        # self.x_hist.append(x)
        self.z_hist.append(obs[0])
        self.z_sum += obs[0]
        self.z_cnt += 1
        # if len(self.x_hist) > 0.5 / self.dt:
        #     self.x_hist = self.x_hist[1:]
        if len(self.z_hist) > 1e4:
            self.z_sum -= self.z_hist[0]
            self.z_cnt -= 1
            self.z_hist = self.z_hist[1:]
        # if dx > 0:
        #     reward += dx / len(self.x_hist) * ( 0.5 / self.dt)
        # if obs[0] < self.z_max - (self.z_max - self.z_init) * 0.32 or \
        #     self.index - self.max_index > 10:
        #     done = True
        #     info["max_height"] = self.z_max
        if self.index - self.max_index > 10:
            done = True
        else:
            if obs[0] > self.z_max:
                self.z_max = min(1.4, obs[0])
                self.max_index = self.index
            if obs[0] > 1:
                self.max_index = self.index
        self._last_obs = obs.copy()
        obs = {}
        i_start = 0
        for i in range(len(self.keys)):
            obs[self.keys[i]] = self._last_obs[i_start:i_start+self.values[i]]
            i_start += self.values[i]
        obs["is_terminal"] = False
        obs["is_first"] = False
        obs["is_last"] = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.model
        data = self.data
        qpos = data.qpos.copy()
        # Reset standing
        # qpos[2] -= np.min(data.xipos[1:,2]) - 0.1
        # Reset laying on the ground
        qpos[2] = 0.15
        qpos[3:7] = [0.707, 0, -0.707, 0]  # quaternion
        qpos[7:] *= 0.5

        qvel = data.qvel.copy()
        # Reset velocities to 0
        qvel[:] = 0.0

        # Apply to simulation
        data.qpos = qpos
        data.qvel = qvel
        mujoco.mj_forward(model, data)

        obs = self.env.unwrapped._get_obs()

        self._last_obs = obs.copy()
        self.x_hist = [qpos[0]]
        self.z_hist = [obs[0]]
        self.z_sum += obs[0]
        self.z_cnt += 1
        self.z_init = obs[0]
        self.obs_init = obs.copy()
        self.z_max = obs[0]
        self.index = 0
        self.max_index = 0
        self.reward = 0

        obs = {}
        i_start = 0
        for i in range(len(self.keys)):
            obs[self.keys[i]] = self._last_obs[i_start:i_start+self.values[i]]
            i_start += self.values[i]
        obs["is_terminal"] = False
        obs["is_first"] = True
        obs["is_last"] = False
            
        return obs

def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "humanoid":
        env = gym.make("Humanoid-v5", max_episode_steps=int(1e10), frame_skip=5,
                    contact_cost_weight=0, forward_reward_weight=0,
                    ctrl_cost_weight=0.1, healthy_reward=0,
                    healthy_z_range=(-1e10, 1e10),
                    render_mode=None)
        env = EnvWrapper(env)
    elif suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
