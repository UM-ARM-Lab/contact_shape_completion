#! /usr/bin/env python
"""
This script takes shapenet (and other?) voxel datasets and compiles ShapenetDatasetSupervisors, which are
metadatasets.

"""
from colorama import Fore

from shape_completion_training.utils import dataset_supervisor
from shape_completion_training.utils.dataset_supervisor import ShapenetDatasetSupervisor, YcbDatasetSupervisor
from argparse import ArgumentParser

shapenet_categories_for = {
    "shapenet_mugs": ['mug'],
    "shapenet_airplanes": ['airplane'],
    "shapenet_tables": ['table'],
    "shapenet_bag": ['bag'],
    "shapenet_all": list(dataset_supervisor.get_shapenet_map().keys())}

ycb_categories_for = {
    "ycb_all": [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "012_strawberry",
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "022_windex_bottle",
        "024_bowl",
        "025_mug",
        "026_sponge",
        "028_skillet_lid",
        "029_plate",
        "030_fork",
        "031_spoon",
        "032_knife",
        "033_spatula",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "038_padlock",
        "040_large_marker",
        "042_adjustable_wrench",
        "043_phillips_screwdriver",
        "044_flat_screwdriver",
        "048_hammer",
        "050_medium_clamp",
        "051_large_clamp",
        "052_extra_large_clamp",
        "053_mini_soccer_ball",
        "054_softball",
        "055_baseball",
        "056_tennis_ball",
        "057_racquetball",
        "058_golf_ball",
        "059_chain",
        "061_foam_brick",
        "062_dice",
        "063-a_marbles",
        "063-b_marbles",
        "065-a_cups",
        "065-b_cups",
        "065-c_cups",
        "065-d_cups",
        "065-e_cups",
        "065-f_cups",
        "065-g_cups",
        "065-h_cups",
        "065-i_cups",
        "065-j_cups",
        "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks",
        "071_nine_hole_peg_test",
        "072-a_toy_airplane",
        "072-b_toy_airplane",
        "072-c_toy_airplane",
        "072-d_toy_airplane",
        "072-e_toy_airplane",
        "073-a_lego_duplo",
        "073-b_lego_duplo",
        "073-c_lego_duplo",
        "073-d_lego_duplo",
        "073-e_lego_duplo",
        "073-f_lego_duplo",
        "073-g_lego_duplo",
        "077_rubiks_cube",
    ]
}


def create_shapenet_only_datasets(overwrite: bool):
    for name, categories in shapenet_categories_for.items():
        ds = ShapenetDatasetSupervisor(name, require_exists=False, load=False)
        if ds.get_save_path().exists():
            print(f"Dataset {name} already exists")
            if not overwrite:
                continue
            print("Overwriting...")
        print(f"Creating dataset {name}...")
        try:
            fps = [dataset_supervisor.get_shapenet_map()[c] for c in categories]
        except KeyError as e:
            print(f"{Fore.RED}Category {e.args[0]} not found for dataset {name}. Skipping this dataset{Fore.RESET}")
            continue
        ds.create_new_dataset(fps)
        ds.save(overwrite=overwrite)
        print(f"Saved Dataset {name}")


def create_ycb_only_dataset(overwrite: bool):
    for name, categories in ycb_categories_for.items():
        ds = YcbDatasetSupervisor(name, require_exists=False, load=False)
        if ds.get_save_path().exists():
            print(f"Dataset {name} already exists")
            if not overwrite:
                continue
            print("Overwriting...")
        print(f"Creating dataset {name}...")
        fps = categories
        ds.create_new_dataset(fps)
        ds.save(overwrite=overwrite)
        print(f"Saved Dataset {name}")


def main():
    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    create_shapenet_only_datasets(args.overwrite)
    create_ycb_only_dataset(args.overwrite)


if __name__ == "__main__":
    main()
