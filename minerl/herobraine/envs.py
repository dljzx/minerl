# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import collections

import gym
from minerl.herobraine import inventory

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs.treechop_specs import Treechop
from minerl.herobraine.env_specs.equip_weapon_specs import EquipWeapon
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.env_specs.navigate_specs import Navigate
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamondShovelEnvSpec
from minerl.herobraine.wrappers import Obfuscated, Vectorized
from minerl.herobraine.env_specs import basalt_specs
from minerl.herobraine.env_specs import env_init
import os

# Must load non-obfuscated envs first!
# Publish.py depends on this order for black-listing streams
MINERL_TREECHOP_V0 = Treechop()

MINERL_NAVIGATE_V0 = Navigate(dense=False, extreme=False)
MINERL_NAVIGATE_EXTREME_V0 = Navigate(dense=False, extreme=True)
MINERL_NAVIGATE_DENSE_V0 = Navigate(dense=True, extreme=False)
MINERL_NAVIGATE_DENSE_EXTREME_V0 = Navigate(dense=True, extreme=True)

MINERL_OBTAIN_DIAMOND_SHOVEL_V0 = ObtainDiamondShovelEnvSpec()

MINERL_EQUIP_WEAPON_V0 = EquipWeapon()
MINERL_HUMAN_SURVIVAL_V0 = HumanSurvival()

MINERL_BASALT_FIND_CAVES_ENV_SPEC = basalt_specs.FindCaveEnvSpec()
MINERL_BASALT_MAKE_WATERFALL_ENV_SPEC = basalt_specs.MakeWaterfallEnvSpec()
MINERL_BASALT_PEN_ANIMALS_VILLAGE_ENV_SPEC = basalt_specs.PenAnimalsVillageEnvSpec()
MINERL_BASALT_VILLAGE_HOUSE_ENV_SPEC = basalt_specs.VillageMakeHouseEnvSpec()

NEW_FLOWER_PLAIN_ENV_SPEC = basalt_specs.FlowerPlainsEnvSpec()

NEW_DESERT_ENV_SPEC = basalt_specs.Desert()

NEW_RIVER_ENV_SPEC = basalt_specs.River()

NEW_JUNGLE_ENV_SPEC = basalt_specs.Jungle()

NEW_BEACH_ENV_SPEC = basalt_specs.Beach()

NEW_SWAMP_ENV_SPEC = basalt_specs.Swamp()

NEW_SAVVAN_ENV_SPEC = basalt_specs.Savanna()

NEW_OCEAN_ENV_SPEC = basalt_specs.Ocean()

NEW_PLAINS_ENV_SPEC = basalt_specs.Plains()

NEW_TAIGA_ENV_SPEC = basalt_specs.Taiga()

NEW_FOREST_ENV_SPEC = basalt_specs.Forest()

NEW_MUSHROOM_ENV_SPEC = basalt_specs.Mushroom()

NEW_ICEPLAINS_ENV_SPEC = basalt_specs.Iceplains()

NEW_MESA_ENV_SPEC = basalt_specs.Mesa()

EXTREME_HILLS = basalt_specs.ExtremeHills()

init_inventory = {'mainhand': {'name': 'iron_sword'}, 'feet': {'name': 'iron_boots'}, 'legs': {'name': 'iron_leggings'}, 'chest': {'name': 'iron_chestplate'}, \
    'head': {'name': 'iron_helmet'}, 'offhand': {'name': 'shield'}}

MINERL_SUMMON_TEST = basalt_specs.Summontest()

# Register the envs.
ENVS = [env for env in locals().values() if isinstance(env, EnvSpec)]
for env in ENVS:
    if env.name not in gym.envs.registry.env_specs:
        env.register()
