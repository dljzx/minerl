import logging
import minerl
import gym
from minerl.human_play_interface.human_play_interface import HumanPlayInterface

import coloredlogs
coloredlogs.install(logging.DEBUG)

ENV_NAMES = [
    "MineRLBasaltFindCave-v0",
    "MineRLBasaltMakeWaterfall-v0",
    "MineRLBasaltCreateVillageAnimalPen-v0",
    "MineRLBasaltBuildVillageHouse-v0", 
    'MineRLTreechop-v0',  # Doesn't work because it isn't a HumanEmbodied environment
]

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[2, 2],
    resolution=[1280, 720],
    # guiscale_range=[1, 1],
    # resolution=[1280, 720],
    cursor_size_range=[16.0, 16.0],
    use_chat_to_control=True
)

def test_human_interface():
    env = gym.make(ENV_NAMES[3])
    env = HumanPlayInterface(env)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step()
    print("Episode done")
    env.close()

if __name__ == '__main__':
    test_human_interface()
