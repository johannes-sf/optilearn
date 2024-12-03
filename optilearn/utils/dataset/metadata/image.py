import imghdr
import random

import numpy as np
from torchvision.datasets.folder import make_dataset, find_classes

from optilearn.utils.dataset.metadata.dto import (
    TrainingMetaData,
)


# ToDo: Write tests
def make_image_training_metadata(
    data_path: str,
    shuffle: bool = True,
    batch_size: int = None,
    class_subset=None,
) -> TrainingMetaData:
    _, label_to_index = find_classes(data_path)

    if class_subset is not None:
        # Making sure label is treated as a string
        _class_subset = [str(class_label) for class_label in class_subset]
        label_to_index = {
            label: index
            for label, index in label_to_index.items()
            if label in _class_subset
        }

        # Reset indices after taking the subset
        label_to_index = {
            label: index for index, label in enumerate(label_to_index.keys())
        }

    labels = sorted(list(label_to_index.keys()))

    # noinspection PyTypeChecker
    file_path_to_class_index: list[tuple] = make_dataset(
        data_path,
        class_to_idx=label_to_index,
        is_valid_file=imghdr.what,
    )

    class_index_to_label: dict = {
        index: label for label, index in label_to_index.items()
    }

    files_to_labels = {}
    file_label_pairs = []
    file_class_index_pairs = []

    for file_path, class_index in file_path_to_class_index:
        class_label = class_index_to_label[class_index]

        file_class_index_pairs.append((file_path, class_index))
        files_to_labels[file_path] = class_label
        file_label_pairs.append((file_path, class_label))

    label_to_files: dict = {label: [] for label in labels}

    for file, label in files_to_labels.items():
        label_to_files[label].append(file)

    _iterator_indices = list(range(0, len(files_to_labels)))

    if shuffle:
        random.shuffle(_iterator_indices)

    index_iterator = _index_generator(_iterator_indices)

    _batch_size = batch_size if batch_size is not None else 1

    total_samples = len(files_to_labels.keys())

    total_iterations = int(np.ceil(total_samples / _batch_size))

    return TrainingMetaData(
        labels_to_files=label_to_files,
        file_label_pairs=file_label_pairs,
        file_class_index_pairs=file_class_index_pairs,
        class_index_to_label=class_index_to_label,
        label_to_class_index=label_to_index,
        labels=labels,
        index_iterator=index_iterator,
        total_iterations=total_iterations,
        total_samples=total_samples,
        batch_size=_batch_size,
        n_labels=len(labels),
    )


def _index_generator(indices):
    for index in indices:
        yield index
