from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class TrainingMetaData:
    """
    Holds meta data for the image loader
    """

    label_to_class_index: dict
    class_index_to_label: dict

    labels: list
    n_labels: int
    batch_size: int
    
    total_iterations: int
    total_samples: int

    index_iterator: Optional[Generator] = None
    labels_to_files: Optional[dict] = None
    file_label_pairs: Optional[list[tuple]] = None
    file_class_index_pairs: Optional[list[tuple]] = None


