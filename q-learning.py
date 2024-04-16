import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0', render_mode="rgb_array")

q_table = np.random.uniform(0,1,(255,env.action_space.n))

def save_gif(rgb_arrays, filename, duration=60):
    frames = []

    for rgb_array in rgb_arrays:
        rgb_array = (rgb_array).astype(np.uint8)
        img = Image.fromarray(rgb_array)
        frames.append(img)

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    # 各値を2個の離散値に変換
    car_pos, car_v = observation
    digitized = [
        np.digitize(car_pos, bins=bins(-1.2, 0.6, 16)),
        np.digitize(car_v, bins=bins(-0.07, 0.07, 16))
                ]
    # 0~255に変換
    return sum([x * (16 ** i) for i, x in enumerate(digitized)])

def decide_action(step, state):
    if np.random.uniform(0,1) > (epsilon / (step+1)):
        action = np.argmax(q_table[state])
    else:
        action = env.action_space.sample()

    return action


def q_learning(state, action, next_state, r):
    alpha = 0.5
    gamma = 0.99
    q_table[state, action] = q_table[state, action] + alpha * (r + gamma * np.max(q_table[next_state]) - q_table[state, action])

frames = []
epsilon = 0.5
rewards = []
max_episodes = 500
max_steps = 300
successed_episodes = 0

for episode in range(max_episodes):
    observation = env.reset()

    state = digitize_state(observation[0])
    
    imgs = [env.render()]
    
    screen = env.render()
    frames.append(screen)
    episode_reward = 0
    for step in range(max_steps):
        
        # epsilon-greedy法により行動を決定
        action = decide_action(step, state)
        # 次の状態を獲得
        next_observation, reward, terminated, _, _ = env.step(action)
        # print("reward: {}, terminated: {}".format(reward, terminated))
        
        # 報酬獲得
        r = next_observation[0] + next_observation[1]
        # if terminated:
        #     print("terminated {} episode in {} step".format(episode+1, step+1))
        #     if step < max_steps-5:
        #         reward = -100
        #     else:
        #         print("{} episode successed".format(episode + 1))
        #         successed_episodes += 1
        #         reward = 1
        # else:
        #     reward = -1

        episode_reward += reward
        # 状態を離散値に変換
        next_state = digitize_state(next_observation)

        # q_learning
        q_learning(state, action, next_state, reward)
        state = next_state
        
        if episode == max_episodes-1:
            imgs.append(env.render())
            screen = env.render()
            frames.append(screen)
            episode_filename= "q-learning" + str(episode) + ".gif"
            save_gif(imgs, episode_filename)
            IImage(filename=episode_filename)

        if terminated or step == max_steps-1:
            rewards.append(episode_reward)
            # print("episode{} finished ofter {} timesteps, total reward {}".format(episode, step+1, total_reward))
            break
print("successed {} / {} episodes".format(successed_episodes, max_episodes))
plt.plot(np.arange(max_episodes), rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
env.close()