import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import numpy as np

env = gym.make("CartPole-v1", render_mode="rgb_array")

def save_gif(rgb_arrays, filename, duration=60):
    frames = []

    for rgb_array in rgb_arrays:
        rgb_array = (rgb_array).astype(np.uint8)
        img = Image.fromarray(rgb_array)
        frames.append(img)

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

q_table = np.random.uniform(0,1,(255,env.action_space.n))
print(q_table)

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    # 各値を4個の離散値に変換
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-4.8, 4.8, 8)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))
                ]
    # 0~255に変換
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

def q_learning(state, action, next_state, alpha, r, gamma):
    q_table[state, action] = q_table[state, action] + alpha * (r + gamma * np.max(q_table[next_state, action]) - q_table[state, action])

frames = []
gamma = 0.99
alpha = 0.1

for episode in range(100):
    observation = env.reset()

    state = digitize_state(observation[0])

    
    imgs = [env.render()]
    
    screen = env.render()
    frames.append(screen)
    done = False
    for state in range(255):
        # decide action base on q_table
        action = np.argmax(q_table[state])
        # get next step env
        next_observation, reward, done, _, _ = env.step(action)

        # convert digitize
        next_state = digitize_state(next_observation)

        # q_learning
        q_learning(state, action, next_state, alpha, reward, gamma)
        # state = next_state
        imgs.append(env.render())
        screen = env.render()
        frames.append(screen)
    if episode == 99:
        episode_filename= "pend_td" + str(episode) + ".gif"
        save_gif(imgs, episode_filename)
        IImage(filename=episode_filename)
env.close()


# import matplotlib.animation as animation
# import matplotlib.pyplot as plt


# plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
# patch = plt.imshow(frames[0])
# plt.axis('off')
# def animate(i):
#     patch.set_data(frames[i])
# anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
# HTML(anim.to_jshtml())