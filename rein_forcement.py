import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class reinforcement_learning:
    def __init__(self, task_name):
        self.env = gym.make(task_name, render_mode="rgb_array")
        self.q_table = np.random.uniform(0,1,(255,self.env.action_space.n))
        self.epsilon = 0.5
        self.alpha = 0.1
        self.gamma = 0.99
        self.max_episodes = 200
        self.max_steps = self.env.spec.max_episode_steps
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

    def digitize_state(self, observations):
        # 各値を4個の離散値に変換
        # cart_pos, cart_v, pole_angle, pole_v = observation
        env_lows = self.env.observation_space.low
        env_highs = self.env.observation_space.high

        digitized = []
        for i, observation in enumerate(observations):
            digitized.append(np.digitize(observation, bins=self.bins(env_lows[i], env_highs[i], 4)))
        # digitized = [
        #     np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, 4)),
        #     np.digitize(cart_v, bins=self.bins(-3.0, 3.0, 4)),
        #     np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, 4)),
        #     np.digitize(pole_v, bins=self.bins(-2.0, 2.0, 4))
        #             ]
        # 0~255に変換
        return sum([x * (4 ** i) for i, x in enumerate(digitized)])
    
    def decide_action(self, step, state):
        if np.random.uniform(0,1) > (self.epsilon / (step+1)):
            action = np.argmax(self.q_table[state])
        else:
            action = self.env.action_space.sample()

        return action
    
    def q_learning(self, state, action, next_state, r):
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (r + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])
    
    def sarsa(self, state, action, next_state, next_action, r):
        gamma = 0.99
        alpha = 0.1
        self.q_table[state, action] = self.q_table[state, action] + alpha * (r + gamma * self.q_table[next_state, next_action] - self.q_table[state, action])
    
    def plot(self, rewards, algorithm, ax):
        
        ax.plot(np.arange(self.max_episodes), rewards)
        ax.xlabel("episode")
        ax.ylabel("reward")
        ax.title(algorithm)
        ax.savefig("./" + algorithm + "/" + algorithm + ".png")
    
    def render(self, imgs, episode_filename):
        for i in tqdm(range(len(imgs))):
            self.save_gif(imgs[:i+1], episode_filename)
            IImage(filename=episode_filename)
    
    def run_Q_learing(self):
        imgs = []
        rewards = []
        for episode in range(self.max_episodes):
            observation = self.env.reset()
            episode_filename = "./q-learning/q-learning" + str(episode + 1) + ".gif"

            state = self.digitize_state(observation[0])
            
            imgs = [self.env.render()]
            
            episode_reward = 0
            for step in range(self.max_steps):
                
                # epsilon-greedy法により行動を決定
                action = self.decide_action(step, state)
                # 次の状態を獲得
                next_observation, reward, terminated, _, _ = self.env.step(action)
                
                # 報酬獲得
                if terminated:
                    if step < self.max_steps-5:
                        reward = -100
                    else:
                        successed_episodes += 1
                        reward = 1
                else:
                    reward = 1

                episode_reward += reward
                # 状態を離散値に変換
                next_state = self.digitize_state(next_observation)

                # q_learning
                self.q_learning(state, action, next_state, reward)
                state = next_state
                
                if episode == self.max_episodes-1:
                    imgs.append(self.env.render())

                if terminated or step == self.max_steps-1:
                    rewards.append(episode_reward)
                    break
        
        self.plot(rewards, "q-learning")
        self.render(imgs, episode_filename)
        self.env.close()
    
    def run_SARSA(self):
        imgs = []
        rewards = []
        for episode in range(self.max_episodes):
            observation = self.env.reset()

            episode_filename = "./sarsa/sarsa" + str(episode + 1) + ".gif"

            state = self.digitize_state(observation[0])
            
            imgs = [self.env.render()]
            
            episode_reward = 0

            action = self.decide_action(1, state)

            for step in range(self.max_steps):

                # 次の状態を獲得
                next_observation, reward, terminated, _, _ = self.env.step(action)
                
                # 報酬獲得
                if terminated:
                    # print("terminated {} episode in {} step".format(episode+1, step+1))
                    if step < self.max_steps-5:
                        reward = -100
                    else:
                        # print("{} episode successed".format(episode + 1))
                        successed_episodes += 1
                        reward = 1
                else:
                    reward = 1

                episode_reward += reward
                # 状態を離散値に変換
                next_state = self.digitize_state(next_observation)

                # 次の行動を決定
                next_action = self.decide_action(step, next_state)

                # sarsa
                self.sarsa(state, action, next_state, next_action, reward)
                state = next_state
                action = next_action
                
                if episode == self.max_episodes-1:
                    imgs.append(self.env.render())
                    # screen = self.env.render()

                if terminated or step == self.max_steps-1:
                    rewards.append(episode_reward)
                    # print("episode{} finished ofter {} timesteps, total reward {}".format(episode, step+1, total_reward))
                    break
        self.plot(rewards, "sarsa")
        self.render(imgs, episode_filename)
        self.env.close()

if __name__ == "__main__":
    r_learning = reinforcement_learning("CartPole-v1")

    r_learning.run_Q_learing()
    r_learning.run_SARSA()
