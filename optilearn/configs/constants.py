from enum import Enum

PROJECT = "your_neptune_work_space/YourProject"
TMP_STORAGE = "./.neptune/neptune_tmp_storage"

IMAGE_FILE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
]


class RunModes(Enum):
    """
    Enum for environment's <run_mode> argument.
    TRAINING & EVAL will both be considered learning modes.
    """

    TRAINING = "training"
    EVAL = "eval"
    INFERENCE = "inference"


class LabelType(Enum):
    """
    Label type of state in image_classification_env
    """
    NORMAL = "normal"
    ONE_HOT = "one_hot"
    SOFT = "soft"
