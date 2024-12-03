import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import SettingWithCopyWarning

from optilearn.utils.dataset.classes import (
    load_label_mapping,
    FILENAME_COL,
    LABEL_COL,
)

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

TRAIN_DIR = "train"
TEST_DIR = "test"


# ToDo: Extend method to work even without train/test sub-directories


def convert_to_torch_format(data_path: str, csv_path: str = None, output_path: str = None) -> None:
    if csv_path is None:
        csv_path = os.path.join(data_path, "labels.csv")

    if output_path is None:
        output_path = os.path.join(data_path, "converted")
        os.makedirs(output_path, exist_ok=True)

    label_mapping = load_label_mapping(path=csv_path)

    for directory in [TRAIN_DIR, TEST_DIR]:
        original_path = os.path.join(data_path, directory)
        if os.path.exists(original_path):
            _distribute_files_in_class_directories(
                path=original_path,
                label_mapping=label_mapping,
                new_path=os.path.join(output_path, directory),
            )


def _distribute_files_in_class_directories(path: str, label_mapping: pd.DataFrame, new_path: str) -> None:
    all_files = list(Path(path).rglob("*"))

    labels = label_mapping[LABEL_COL].unique().tolist()
    for label in labels:
        os.makedirs(os.path.join(new_path, label), exist_ok=True)

    for file in all_files:
        file_name = file.name.split(".")[0]

        try:
            label = _get_label(file_name, label_mapping)
        except KeyError as label_not_found_error:
            warnings.warn(str(label_not_found_error))
            continue

        new_file = os.path.join(new_path, label, file.name)

        shutil.copy2(file, new_file)


def _get_label(file_name: str, label_mapping: pd.DataFrame) -> str:
    label_candidates = label_mapping.loc[label_mapping[FILENAME_COL] == file_name, LABEL_COL]

    if not len(label_candidates):
        raise KeyError(f"File {file_name} not found in label mapping.")

    return label_candidates.values[0]
