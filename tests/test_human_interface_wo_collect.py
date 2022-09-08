import logging
import minerl
import gym
from minerl.human_play_interface.human_play_interface import HumanPlayInterface

import os
import cv2
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
    use_chat_to_control=False
)

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.env_specs.human_controls import SimpleHumanEmbodimentEnvSpec
from minerl.herobraine.inventory import InventoryItem

save_path = 'results/'

def test_human_interface():
    # env = gym.make(ENV_NAMES[3])
    env = HumanSurvival(**ENV_KWARGS).make()
    # env = SimpleHumanEmbodimentEnvSpec(**ENV_KWARGS).make()
    env = HumanPlayInterface(env)
    env.reset()
    done = False
    i = 0
    
    # use cmd api to set inventory
    inventory_list = [InventoryItem(name="crafting_table", slot=0, quantity=1, variant=0),
                      InventoryItem(name="diamond", slot=1, quantity=64, variant=0)]
    env.env._cmd_executor.set_inventory(inventory_list)

    # update chat in action
    action = env.action_space.noop()
    print(action)
    action['chat'] = '/give @p diamond 3'
    env.env.step(action)
    
    while not done:
        obs, reward, done, info = env.step()
        
    print("Episode done")
    env.close()

if __name__ == '__main__':
    test_human_interface()
