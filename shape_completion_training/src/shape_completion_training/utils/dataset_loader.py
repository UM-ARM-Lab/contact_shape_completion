from colorama import Fore

from shape_completion_training.utils.dataset_supervisor import ShapenetDatasetSupervisor, YcbDatasetSupervisor
from shape_completion_training.utils.dataset_supervisor_AAB import AabDatasetSupervisor


def get_dataset_supervisor(dataset: str):
    if dataset.startswith("shapenet"):
        print(f"{Fore.GREEN}Loading Shapenet Dataset {dataset}{Fore.RESET}")
        return ShapenetDatasetSupervisor(dataset)
    elif dataset.startswith("ycb"):
        print(f"{Fore.GREEN}Loading YCB Dataset {dataset}{Fore.RESET}")
        return YcbDatasetSupervisor(dataset)
    elif dataset.startswith("aab"):
        print(f"{Fore.GREEN}Loading AAB Dataset {dataset}{Fore.RESET}")
        return AabDatasetSupervisor(dataset)
    raise RuntimeError(f"Error: Unknown dataset {dataset}")