# A simple pyglet app which controls the MineRL env,
# showing human the MineRL image and passing game controls
# to MineRL
# Intended for quick data collection without hassle or
# human corrections (agent plays but human can take over).
import os
from typing import List, Optional, Dict, Any
import time
from collections import defaultdict

import random
import math
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
from minerl.herobraine import inventory
from minerl.herobraine.cmd_executor import CMDExecutor
from minerl.herobraine.inventory import InventoryItem
from omegaconf import OmegaConf
import importlib_resources

def _resource_file_path(fname) -> str:
    with importlib_resources.path("minerl.env", fname) as p:
        return str(p)

# Mapping from MineRL action space names to pyglet keys
MINERL_ACTION_TO_KEYBOARD = {
    "ESC":       key.ESCAPE, # Used in BASALT to end the episode
    "attack":    pyglet.window.mouse.LEFT,
    "back":      key.S,
    "drop":      key.Q,
    "forward":   key.W,
    "hotbar.1":  key._1,
    "hotbar.2":  key._2,
    "hotbar.3":  key._3,
    "hotbar.4":  key._4,
    "hotbar.5":  key._5,
    "hotbar.6":  key._6,
    "hotbar.7":  key._7,
    "hotbar.8":  key._8,
    "hotbar.9":  key._9,
    "inventory": key.E,
    "jump":      key.SPACE,
    "left":      key.A,
    "pickItem":  pyglet.window.mouse.MIDDLE,
    "right":     key.D,
    "sneak":     key.LSHIFT,
    "sprint":    key.LCTRL,
    "swapHands": key.F,
    "use":       pyglet.window.mouse.RIGHT
}

KEYBOARD_TO_MINERL_ACTION = {v: k for k, v in MINERL_ACTION_TO_KEYBOARD.items()}

def _parse_inventory_dict(inv_dict: dict[str, dict]) -> list[InventoryItem]:
    return [InventoryItem(slot=k, **v) for k, v in inv_dict.items()]

# Camera actions are in degrees, while mouse movement is in pixels
# Multiply mouse speed by some arbitrary multiplier
MOUSE_MULTIPLIER = 0.1

MINERL_FPS = 20
MINERL_FRAME_TIME = 1 / MINERL_FPS
CUS_TASKS_SPECS = OmegaConf.load(_resource_file_path("custom_tasks_specs.yaml"))
class HumanPlayInterface(gym.Wrapper):
    def __init__(self, minerl_env, task_id=None):
        super().__init__(minerl_env)
        self._validate_minerl_env(minerl_env)
        self.env = minerl_env
        print(self.env.__class__)
        pov_shape = self.env.observation_space["pov"].shape
        self.window = pyglet.window.Window(
            width=pov_shape[1],
            height=pov_shape[0],
            vsync=False,
            resizable=False
        )
        self._cmd_exe = CMDExecutor(self, False)
        self.clock = pyglet.clock.get_default()
        self.pressed_keys = defaultdict(lambda: False)
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactive = self._on_window_deactivate
        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()

        self.needtp = False
        self.biome = None

        self.last_pov = None
        self.last_mouse_delta = [0, 0]
        self.task_id = task_id
        self.inventory =None
        self.mobs =None
        self.blocks =None
        self.specified_biome = None
        self.loot = False

        if self.task_id != None:
            task_specs = CUS_TASKS_SPECS[self.task_id].copy()
            if 'needtp'in task_specs.keys():
                self.needtp = True
                self.biome = task_specs['specified_biome']

            if 'initial_loot'in task_specs.keys():
                self.loot = True
                self.loot_items = task_specs['initial_loot']
                self.loot_positions = task_specs['loot_position']

            if 'initial_inventory' in task_specs.keys():
                self.inventory = task_specs['initial_inventory']
                print(self.inventory)

            if 'initial_mobs' in task_specs.keys():
                self.mobs = task_specs['initial_mobs']
                print(self.mobs)
                self.mobs_max_dis = task_specs['mobs_max_dis']
                self.mobs_min_dis = task_specs['mobs_min_dis']

            if 'initial_blocks' in task_specs.keys():
                self.blocks = task_specs['initial_blocks']
                print(self.blocks)
                self.blocks_max_dis = task_specs['blocks_max_dis']
                self.blocks_min_dis = task_specs['blocks_min_dis']
            
            if 'specified_biome' in task_specs.keys():
                self.specified_biome = task_specs['specified_biome']
                print(self.specified_biome)

        self.window.clear()
        self._show_message("Waiting for reset.")

    def _on_key_press(self, symbol, modifiers):
        self.pressed_keys[symbol] = True

    def _on_key_release(self, symbol, modifiers):
        self.pressed_keys[symbol] = False

    def _on_mouse_press(self, x, y, button, modifiers):
        self.pressed_keys[button] = True

    def _on_mouse_release(self, x, y, button, modifiers):
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, button, modifier):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def teleport_safe_position(self, position):
        # teleport the surface of z, x.
            
        
        # close fall damage
        action = self.env.action_space.noop()
        action['chat'] = f'/gamerule fallDamage false'
        self.env.step(action)
        
        # rollout trajectory
        for _ in range(5):
            self.env.step(self.env.action_space.noop())
        
        # tp to highest position
        position = ' '.join(position.split(' ')[0:1]+['150']+position.split(' ')[-1:])
        action = self.env.action_space.noop()
        action['chat'] = f'/teleport @s {position}'
        self.env.step(action)
        
        # rollout trajectory
        for _ in range(100):
            self.env.step(self.env.action_space.noop())
        
        # open fall damage
        action = self.env.action_space.noop()
        action['chat'] = f'/gamerule fallDamage true'
        self.env.step(action)
        
        # rollout trajectory
        for _ in range(5):
            self.env.step(self.env.action_space.noop())
    

    def spawn_biome_generator(self, specific_biome='plains'):
        
        # log path
        logdir = os.environ.get('MALMO_MINECRAFT_OUTPUT_LOGDIR', '.')
        log_dir = os.path.join(logdir, 'logs', f'mc_{self.env.instances[0]._target_port - 9000}.log')
        
        # find position
        action = self.env.action_space.noop()
        action['chat'] = f'/locatebiome {specific_biome}'
        self.env.step(action)
        
        # rollout trajectory
        for _ in range(10):
            self.env.step(self.env.action_space.noop())
            
        # tp to target biome
        import re
        with open(log_dir, 'r') as f:
            lines = f.readlines()
            sample = [l for l in lines if '[CHAT] The nearest' in l][0]
        position = ' '.join(re.search('at.*\[.*\]', sample).group().replace(',', '').replace('[', '').replace(']', '').split(' ')[1:])
        action = self.env.action_space.noop()
        
        # # tp to safe location
        self.teleport_safe_position(position)
        # action['chat'] = f'/teleport @s {position}'
        # action['chat'] = f'/execute in overworld run tp @s {position}'
        # env.step(action)
        # # rollout trajectory
        # for _ in range(10):
        #     env.step(env.action_space.noop())

    def _validate_minerl_env(self, minerl_env):
        """Make sure we have a valid MineRL environment. Raises if not."""
        # Make sure action has right items
        remaining_buttons = set(list(MINERL_ACTION_TO_KEYBOARD.keys()))
        remaining_buttons.add("camera")
        if self.env.task.use_chat_to_control:
            remaining_buttons.add("chat")
        for action_name, action_space in minerl_env.action_space.spaces.items():
            if action_name not in remaining_buttons:
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} is not supported.")
            elif (not isinstance(action_space, spaces.Discrete) or action_space.n != 2) and action_name not in ["camera", "chat"]:
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} had space {action_space}. Only Discrete(2) is supported.")
            remaining_buttons.remove(action_name)
        if len(remaining_buttons) > 0:
            raise RuntimeError(f"Invalid MineRL action space: did not contain actions {remaining_buttons}")

        obs_space = minerl_env.observation_space
        if not isinstance(obs_space, spaces.Dict) or "pov" not in obs_space.spaces:
            raise RuntimeError("Invalid MineRL observation space: observation space must contain POV observation.")

    def _update_image(self, arr):
        self.window.switch_to()
        # Based on scaled_image_display.py
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, 0)
        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action = {
            name: int(self.pressed_keys[key] if key is not None else None) for name, key in MINERL_ACTION_TO_KEYBOARD.items()
        }

        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]
        return action

    def _show_message(self, text):
        label = pyglet.text.Label(
            text,
            font_size=32,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x='center',
            anchor_y='center'
        )
        label.draw()
        self.window.flip()

    def reset(self):
        self.window.clear()
        self._show_message("Resetting environment...")
        obs = self.env.reset()

        if self.needtp:
            self.spawn_biome_generator(specific_biome=self.specified_biome)
        if self.inventory != None:
            obs = self._after_sim_reset_inventory(self.inventory)
        if self.mobs != None:
            obs = self._after_sim_reset_mobs(self.mobs)
        if self.blocks != None:
            obs = self._after_sim_reset_blocks(self.blocks)
        if self.loot:
            obs = self._after_sim_reset_loot(self.loot_items,self.loot_positions)
        
        # obs, info = self._after_sim_reset_hook(obs,self.mobs,self.blocks)
        self._update_image(obs["pov"])
        self.clock.tick()
        return obs

    def _after_sim_reset_inventory(self, inventory_list: InventoryItem=None):
        if inventory_list != None:
            inventory = _parse_inventory_dict(
                inventory_list
            )
            obs, _, _, info = self._cmd_exe.set_inventory(
                inventory
            )
            print('finish resetting inventory')
        return obs

    def _after_sim_locate_biome(self, biome: str =None):
        if biome != None:
            
            obs, _, _, info = self._cmd_exe.locate_biome(
                biome
            )
            print('finish locate biome')
        return obs

    def _after_sim_reset_loot(self, loot_items: List[str]=None, loot_positions: List[List[int]]=None):
        if loot_items != None:
            
            obs, _, _, info = self._cmd_exe.spawn_items(
                loot_items, [loot_positions]
            )
            print('finish resetting loot')
        return obs

    def _after_sim_reset_mobs(self, mobs: List[str]=None):
        if mobs != None:
            dis = math.ceil((self.mobs_max_dis / math.sqrt(2)))
            print('dis:',dis)
            mobs_position = []
            for _ in mobs:
                x, z = 0.0 , 0.0
                while math.sqrt(x*x+z*z)<self.mobs_min_dis:
                    x = random.uniform(-dis,dis)
                    z = random.uniform(-dis,dis)
                mobs_position.append([x,1,z])
            
            print('mobs_position',mobs_position)
            obs, _, _, info = self._cmd_exe.spawn_mobs(
                mobs, mobs_position
            )
            print('finish summon mobs')
        return obs

    def _after_sim_reset_blocks(self, blocks: List[str]=None):
        if blocks != None:
            dis = math.ceil((self.blocks_max_dis / math.sqrt(2)))
            print('dis:',dis)
            blocks_position = []
            for _ in blocks:
                x, z = 0.0 , 0.0
                while math.sqrt(x*x+z*z)<self.blocks_min_dis:
                    x = random.uniform(-dis,dis)
                    z = random.uniform(-dis,dis)
                blocks_position.append([x,0,z])
            
            print('blocks_position',blocks_position)
            obs, _, _, info = self._cmd_exe.set_block(
                blocks, blocks_position
            )
            print('finish setting blocks')
        return obs

    def _after_sim_reset_hook(
        self, reset_obs: Dict[str, Any], mobs: List[str]=None, blocks: List[str]=None):
        mobs_rel_positions = []
        if mobs != None:
            # for i in range(len(self._initial_mobs)):
            #     mobs_rel_position = self._mob_spawn_range_space.sample()
            #     if self.min_spawn_range != None:
            #         dis = mobs_rel_position[0]**2 + mobs_rel_position[2]**2
            #         print(dis)
            #         while dis < self.min_spawn_range**2:
            #             print('retry')
            #             mobs_rel_position = self._mob_spawn_range_space.sample()
            #             dis = mobs_rel_position[0]**2 + mobs_rel_position[1]**2
            #     mobs_rel_positions.append(mobs_rel_position)
            #     print('mobs_rel_position: ',mobs_rel_position)

            # print('mobs_rel_positions: ',mobs_rel_positions)
            obs, _, _, info = self._cmd_exe.spawn_mobs(
                mobs, [[0,1,1]]
            )
            print('finish summon mobs')

        blocks_rel_positions = []
        if blocks != None:
            # for i in range(len(self._initial_blocks)):
                # blocks_rel_position = self._block_set_range_space.sample()
                # if self.min_block_range != None:
                #     dis = blocks_rel_position[0]**2 + blocks_rel_position[2]**2
                #     print(dis)
                #     while dis < self.min_block_range**2:
                #         print('retry')
                #         blocks_rel_position = self._block_set_range_space.sample()
                #         dis = blocks_rel_position[0]**2 + blocks_rel_position[1]**2
                # blocks_rel_positions.append(blocks_rel_position)
                # print('blocks_rel_position: ',blocks_rel_position)

            # print('blocks_rel_positions: ',blocks_rel_positions)
            # obs, _, _, info = self.env.set_block(
            #     self._initial_blocks, blocks_rel_positions
            # )
            
            obs, _, _, info = self._cmd_exe.set_block(
                blocks, [[0,1,4]]
            )
            print('finish setting blocks')

        return obs, info

    def step(self, action: Optional[dict] = None, override_if_human_input: bool = False):
        """
        Step environment for one frame.

        If `action` is not None, assume it is a valid action and pass it to the environment.
        Otherwise read action from player (current keyboard/mouse state).

        If `override_if_human_input` is True, execeute action from the human player if they
        press any button or move mouse.

        The executed action will be added to the info dict as "taken_action".
        """
        time_to_sleep = MINERL_FRAME_TIME - self.clock.tick()
        if time_to_sleep > 0:
            self.clock.sleep(int(time_to_sleep * 1000))
        if not action or override_if_human_input:
            self.window.dispatch_events()
            human_action = self._get_human_action()
            if override_if_human_input:
                if any(v != 0 if name != "camera" else (v[0] != 0 or v[1] != 0) for name, v in human_action.items()):
                    action = human_action
            else:
                action = human_action
        if self.env.task.use_chat_to_control:
            # action['chat'] = '/give @p diamond 3'
            action['chat'] = ''

        obs, reward, done, info = self.env.step(action)
        self._update_image(obs["pov"])

        if done:
            self._show_message("Episode done.")

        info["taken_action"] = action
        return obs, reward, done, info
