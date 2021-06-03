from colorama import Fore

def get_noiseless_params():
    noiseless = {
        'translation_pixel_range_x': 0,
        'translation_pixel_range_y': 0,
        'translation_pixel_range_z': 0,
        'apply_slit_occlusion': False,
        'apply_depth_sensor_noise': False,
    }
    return noiseless


def get_visualization_params():
    default_params = {
        'translation_pixel_range_x': 10,
        'translation_pixel_range_y': 10,
        'translation_pixel_range_z': 10,
        'apply_slit_occlusion': True,
        'apply_depth_sensor_noise': False,
    }
    return default_params


def get_default_params(group_name=None):
    default_params = {
        'translation_pixel_range_x': 10,
        'translation_pixel_range_y': 10,
        'translation_pixel_range_z': 10,
        'simulate_partial_completion': False,
        'simulate_random_partial_completion': False,
        # 'network': 'VoxelCNN',
        # 'network': 'VAE_GAN',
        # 'network': 'Augmented_VAE',
        # 'network': 'Conditional_VCNN',
        # 'network': 'NormalizingAE',
        'learning_rate': 1e-3,
        'batch_size': 16,
        'dataset': 'shapenet_mugs',
        'apply_slit_occlusion': False,
        'apply_depth_sensor_noise': False,
    }

    if group_name is None:
        print(Fore.YELLOW + "Loading default params with no group name" + Fore.RESET)
        return default_params

    group_defaults = {
        "PSSNet":
            {
                'num_latent_layers': 200,
                'flow': 'Flow/July_02_10-47-22_d8d84f5d65',
                'network': 'PSSNet',
                'use_flow_during_inference': False
            },
        "PSSNet_shapenet_mugs":
            {
                'num_latent_layers': 200,
                'flow': 'Flow_shapenet_mugs/June_03_10-42-08_ad03459844',
                'network': 'PSSNet',
                'use_flow_during_inference': False,
                'apply_depth_sensor_noise': True,
                'dataset': 'shapenet_mugs',
            },
        "PSSNet_shapenet_all":
            {
                'num_latent_layers': 200,
                'flow': 'Flow_shapenet_all/May_24_15-14-58_28829eda5b',
                'network': 'PSSNet',
                'use_flow_during_inference': False,
                'apply_depth_sensor_noise': True,
                'dataset': 'shapenet_all',
            },
        "NormalizingAE":
            {
                'num_latent_layers': 200,
                'flow': 'Flow/July_02_10-47-22_d8d84f5d65',
                'network': 'NormalizingAE',
                'use_flow_during_inference': False
            },
        "VAE":
            {
                'num_latent_layers': 200,
                'network': 'VAE'
            },
        "VAE_GAN":
            {
                'num_latent_layers': 200,
                'network': 'VAE_GAN',
                'learning_rate': 0.0001,
                'discriminator_learning_rate': 0.00005,
            },
        "Flow":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 10,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
            },
        "Flow_shapenet_all":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 10,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'dataset': 'shapenet_all'
            },
        "Flow_shapenet_mugs":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 10,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'dataset': 'shapenet_mugs'
            },
        "FlowYCB":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 20,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'dataset': 'ycb_all',
            },
        "Flow_aab":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 15,
                'translation_pixel_range_z': 15,
                'dataset': 'aab',
            },
        "3D_rec_gan":
            {
                'batch_size': 4,
                'network': '3D_rec_gan',
                "learning_rate": 0.0001,
                "gan_learning_rate": 0.00005,
                "num_latent_layers": 2000,
                "is_u_connected": True,
            },
        "PSSNet_YCB":
            {
                'num_latent_layers': 200,
                'flow': 'FlowYCB/May_24_13-31-45_85d3ccb8ca',
                'network': 'PSSNet',
                'use_flow_during_inference': False,
                'dataset': 'ycb_all',
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'apply_slit_occlusion': True,
                'apply_depth_sensor_noise': True
            },
        "PSSNet_aab":
            {
                'num_latent_layers': 200,
                'flow': 'Flow_aab/May_26_20-05-11_eed18af77d',
                'network': 'PSSNet',
                'use_flow_during_inference': False,
                'dataset': 'aab',
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'apply_slit_occlusion': True,
                'apply_depth_sensor_noise': True
            },
        "NormalizingAE_YCB":
            {
                'num_latent_layers': 200,
                'flow': 'FlowYCB/July_23_16-35-55_7d12d68bee',
                'network': 'NormalizingAE',
                'use_flow_during_inference': False,
                'dataset': 'ycb_all',
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'apply_slit_occlusion': True,
            },
        "VAE_YCB":
            {
                'num_latent_layers': 200,
                'network': 'VAE',
                'dataset': 'ycb_all',
                'apply_slit_occlusion': True,
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
            },
        "VAE_GAN_YCB":
            {
                'num_latent_layers': 200,
                'network': 'VAE_GAN',
                'learning_rate': 0.0001,
                'discriminator_learning_rate': 0.00005,
                'dataset': 'ycb_all',
                'apply_slit_occlusion': True,
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
            },
        "3D_rec_gan_YCB":
            {
                'batch_size': 4,
                'network': '3D_rec_gan',
                "learning_rate": 0.0001,
                "gan_learning_rate": 0.00005,
                "num_latent_layers": 2000,
                "is_u_connected": True,
                'dataset': 'ycb_all',
                'apply_slit_occlusion': True,
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
            },
        "NormalizingAE_YCB_noise":
            {
                'num_latent_layers': 200,
                'flow': 'FlowYCB/July_23_16-35-55_7d12d68bee',
                'network': 'NormalizingAE',
                'use_flow_during_inference': False,
                'dataset': 'ycb_all',
                'translation_pixel_range_x': 15,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
                'apply_slit_occlusion': True,
                'apply_depth_sensor_noise': True,
            },
        "NormalizingAE_noise":
            {
                'num_latent_layers': 200,
                'flow': 'Flow/July_02_10-47-22_d8d84f5d65',
                'network': 'NormalizingAE',
                'use_flow_during_inference': False,
                'apply_depth_sensor_noise': True,
            },
        "PSSNet_table":
            {
                'num_latent_layers': 200,
                'flow': 'Flow/November_29_16-22-59_5a71a4dcac',
                'network': 'PSSNet',
                'use_flow_during_inference': False
            },
    }

    if group_name not in group_defaults:
        print("Group name {} not in group defaults. Add to default_params. "
              "Current groups are {}".format(group_name,
                                             group_defaults.keys()))
        raise Exception("Group name {} not in group defaults.".format(group_name))

    default_params.update(group_defaults[group_name])
    return default_params
