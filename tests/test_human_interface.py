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
    "Summontest-v0",
]

NEW_NAMES = [
    "Summontest-v0",
    "FlowerPlains-v0",   # sunflower-plains doesn't work
    "ExtremeHills-v0",
    "Desert-v0",
    "Beach-v0",
    "Ocean-v0",
    "Savanna-v0",
    "Taiga-v0",
    "Jungle-v0",
    "River-v0",
    "Swamp-v0",
    "Forest-v0",
    "Mesa-v0",
    "Mushroom-v0",
    "Iceplains-v0",
]

VALID_NAMES = [
    "ExtremeHills-v0",
    "Beach-v0",
    "Ocean-v0",
    "Taiga-v0",
    "River-v0",
    "Swamp-v0",
    "Forest-v0",
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
    env = gym.make(NEW_NAMES[-1])
    # env = gym.make(NEW_NAMES[1])
    env = HumanPlayInterface(env, "custom_combat_8")
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step()
    print("Episode done")
    env.close()

if __name__ == '__main__':
    test_human_interface()
