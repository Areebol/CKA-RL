import gymnasium as gym

env = gym.make("CartPole-v1")  # 使用一个简单的CPU环境
state = env.reset()
for _ in range(10):
    state, reward, done, truncated, info = env.step(env.action_space.sample())
    if done:
        state = env.reset()
env.close()
print("Test completed successfully.")
