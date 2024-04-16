import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="rgb_array")

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
    # 各値を4個の離散値に変換
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))
                ]
    # 0~255に変換
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

def decide_action(step, state):
    if np.random.uniform(0,1) > (epsilon / (step+1)):
        action = np.argmax(q_table[state])
    else:
        action = env.action_space.sample()

    return action

def sarsa(state, action, next_state, next_action, r):
    gamma = 0.99
    alpha = 0.1
    q_table[state, action] = q_table[state, action] + alpha * (r + gamma * q_table[next_state, next_action] - q_table[state, action])

frames = []
epsilon = 0.5
rewards = []
max_episodes = 500
max_steps = 200
successed_episodes = 0

for episode in range(max_episodes):
    observation = env.reset()

    state = digitize_state(observation[0])
    
    imgs = [env.render()]
    
    screen = env.render()
    frames.append(screen)
    episode_reward = 0

    action = decide_action(1, state)

    for step in range(max_steps):

        # 次の状態を獲得
        next_observation, reward, terminated, _, _ = env.step(action)
        
        # 報酬獲得
        if terminated:
            print("terminated {} episode in {} step".format(episode+1, step+1))
            if step < max_steps-5:
                reward = -100
            else:
                print("{} episode successed".format(episode + 1))
                successed_episodes += 1
                reward = 1
        else:
            reward = 1

        episode_reward += reward
        # 状態を離散値に変換
        next_state = digitize_state(next_observation)

        # 次の行動を決定
        next_action = decide_action(step, next_state)

        # sarsa
        sarsa(state, action, next_state, next_action, reward)
        state = next_state
        action = next_action
        
        if episode == max_episodes-1:
            imgs.append(env.render())
            screen = env.render()
            frames.append(screen)
            episode_filename= "sarsa" + str(episode) + ".gif"
            save_gif(imgs, episode_filename)
            IImage(filename=episode_filename)

        if terminated or step == max_steps-1:
            rewards.append(episode_reward)
            # print("episode{} finished ofter {} timesteps, total reward {}".format(episode, step+1, total_reward))
            break
plt.plot(np.arange(max_episodes), rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
env.close()