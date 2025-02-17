import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class reinforcement_learning:
    def __init__(self):
        self.epsilon = 0.5
        self.alpha = 0.5
        self.gamma = 0.99
        self.max_episodes = 2000
        self.successed_episode = 0
    
    def save_gif(self, rgb_arrays, filename, duration=60):
        frames = []
        for rgb_array in rgb_arrays:
            rgb_array = (rgb_array).astype(np.uint8)
            img = Image.fromarray(rgb_array)
            frames.append(img)

        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observations, env):
        # 各値を4個の離散値に変換
        # cart_pos, cart_v, pole_angle, pole_v = observation
        env_lows = env.observation_space.low
        env_highs = env.observation_space.high

        digitized = []
        cart_pos = observations[0]
        cart_v = observations[1]
        pole_angle = observations[2]
        pole_v = observations[3]
        # for i, observation in enumerate(observations):
        #     digitized.append(np.digitize(observation, bins=self.bins(env_lows[i], env_highs[i], 4)))
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, 4)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, 4)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, 4)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, 4))
                    ]
        # 0~255に変換
        return sum([x * (4 ** i) for i, x in enumerate(digitized)])
    
    def decide_action(self, episode, state, q_table, env):
        if np.random.uniform(0,1) > (self.epsilon / (episode+1)):
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        return action
    
    def q_learning(self, q_table, state, action, next_state, r):
        q_table[state, action] = q_table[state, action] + self.alpha * (r + self.gamma * np.max(q_table[next_state]) - q_table[state, action])
    
    def sarsa(self, q_table, state, action, next_state, next_action, r):
        gamma = 0.99
        alpha = 0.1
        q_table[state, action] = q_table[state, action] + alpha * (r + gamma * q_table[next_state, next_action] - q_table[state, action])
    
    def plot(self, rewards, rewards_mean, rewards_median, algorithm):
        
        plt.plot(np.arange(self.max_episodes), rewards, label="episode_reward")
        plt.plot(np.arange(self.max_episodes), rewards_mean, label="episode_reward_mean")
        plt.plot(np.arange(self.max_episodes), rewards_median, label="episode_reward_median")
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend()
        plt.title(algorithm)
        plt.savefig("./" + algorithm + "/" + algorithm + ".png")
        plt.show()
    
    def render(self, imgs, episode_filename):
        for i in tqdm(range(len(imgs))):
            self.save_gif(imgs[:i+1], episode_filename)
            IImage(filename=episode_filename)
    
    def run_Q_learing(self, task):
        env = gym.make(task, render_mode="rgb_array")
        q_table = np.random.uniform(0,1,(255,env.action_space.n))
        max_steps = env.spec.max_episode_steps
        rewards = []
        rewards_mean = []
        rewards_median = []
        successed_episodes = 0
        
        for episode in range(self.max_episodes):
            observation = env.reset()

            state = self.digitize_state(observation[0], env)\

            episode_reward = 0
            for step in range(max_steps):
                
                # epsilon-greedy法により行動を決定
                action = self.decide_action(episode, state, q_table, env)
                # 次の状態を獲得
                next_observation, reward, terminated, truncated, _ = env.step(action)
                
                # 報酬獲得
                if terminated:
                    if step != max_steps-1:
                        reward = -100
                    else:
                        successed_episodes += 1
                        reward = 1
                else:
                    reward = 1

                episode_reward += reward

                # 状態を離散値に変換
                next_state = self.digitize_state(next_observation, env)

                # q_learning
                self.q_learning(q_table, state, action, next_state, reward)
                state = next_state
                
                if terminated or truncated or step == max_steps-1:
                    env.reset()
                    rewards.append(episode_reward)
                    rewards_mean.append(np.mean(rewards))
                    rewards_median.append(np.median(rewards))
                    break
        
        self.plot(rewards, rewards_mean, rewards_median, "q-learning")
        env.close()
    
    def run_SARSA(self, task):
        env = gym.make(task, render_mode="rgb_array")
        q_table = np.random.uniform(0,1,(255,env.action_space.n))
        rewards = []
        rewards_mean = []
        rewards_median = []
        successed_episodes = 0
        max_steps = env.spec.max_episode_steps
        for episode in range(self.max_episodes):
            observation = env.reset()

            state = self.digitize_state(observation[0], env)
            
            episode_reward = 0

            action = self.decide_action(episode, state, q_table, env)

            for step in range(max_steps):

                # 次の状態を獲得
                next_observation, reward, terminated, truncated, _ = env.step(action)
                
                # 報酬獲得
                if terminated:
                    if step < max_steps-5:
                        reward = -100
                    else:
                        successed_episodes += 1
                        reward = 1
                else:
                    reward = 1

                episode_reward += reward
                # 状態を離散値に変換
                next_state = self.digitize_state(next_observation, env)

                # 次の行動を決定
                next_action = self.decide_action(episode, next_state, q_table, env)

                # sarsa
                self.sarsa(q_table, state, action, next_state, next_action, reward)
                state = next_state
                action = next_action

                if terminated or truncated or step == max_steps-1:
                    rewards.append(episode_reward)
                    rewards_mean.append(np.mean(rewards))
                    rewards_median.append(np.median(rewards))
                    env.reset()
                    break
        self.plot(rewards, rewards_mean, rewards_median, "sarsa")
        env.close()

if __name__ == "__main__":
    task = "CartPole-v1"
    r_learning = reinforcement_learning()

    r_learning.run_Q_learing(task)
    r_learning.run_SARSA(task)

