from typing import Callable

import torch
from line_profiler import profile
from torch.utils.data import DataLoader
from torchvision import transforms

from optilearn.configs.constants import LabelType
from optilearn.environments.data_loaders.abstract_data_loader import AbstractDataLoader
from optilearn.environments.data_loaders.filtered_image_folder import FilteredImageFolder
from optilearn.environments.responses import DataLoaderResponse, LearningImageLoaderResponse
from optilearn.utils.dataset.label_transformations import transform_labels
from optilearn.utils.dataset.metadata.dto import TrainingMetaData


class SimpleImageLoader(AbstractDataLoader):
    """
    Torch based generic image data-loader. 
    """
    @profile
    def __init__(
        self,
        data_path: str = None,
        transform_pipeline=None,
        learning_mode: bool = True,
        batch_size: int = 64,
        shuffle=True,
        class_subset: list[str] = None,
        label_type: LabelType = LabelType.ONE_HOT,
        img_size: tuple = (255, 255),
        num_workers: int = 8,
        device: str = "cpu",
        **kwargs,
    ):
        super(SimpleImageLoader, self).__init__(
            learning_mode=learning_mode,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )
        self.data_path = data_path
        self.image_size = img_size
        self.label_type = label_type
        self.transformation_pipeline = transform_pipeline
        self.num_workers = num_workers
        self.device = device
        self.learning_mode = learning_mode

        self.norm_mean = kwargs.get("norm_means", [0.5, 0.5, 0.5])
        self.norm_std = kwargs.get("norm_std", [0.5, 0.5, 0.5])
        self.rotation = kwargs.get("rotation", 30)

        if class_subset is not None:
            self.class_subset = [str(label) for label in class_subset]
        else:
            self.class_subset = None

        # todo move outside
        if learning_mode:
            transform_pipeline: Callable = torch.nn.Sequential(
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(self.rotation),
            )
        else:
            transform_pipeline: Callable = torch.nn.Sequential(
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            )
        augmentation_transforms = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transform_pipeline,
            ]
        )

        self.transformation_pipeline = transform_pipeline

        self.image_dataset = FilteredImageFolder(
            root=data_path,
            transform=augmentation_transforms,
            class_subset=self.class_subset,
        )

        self.data_loader = DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        self._iterator = self.data_generator()
        self.meta_data = self._make_meta_data()

    @profile
    def data_generator(self):
        for images, labels in self.data_loader:
            yield images, labels

    @profile
    def _make_meta_data(self):
        return TrainingMetaData(
            label_to_class_index=self.image_dataset.class_to_idx,
            class_index_to_label={index: label for label, index in self.image_dataset.class_to_idx.items()},
            labels=self.image_dataset.classes,
            n_labels=len(self.image_dataset.classes),
            batch_size=self.batch_size,
            total_iterations=len(self.data_loader),
            total_samples=len(self.image_dataset.samples),
        )

    @profile
    def step(self) -> DataLoaderResponse:
        images, class_indices = next(self.iterator)

        images = images.to(self.device)
        class_indices = class_indices.to(self.device)

        sample = LearningImageLoaderResponse(
            state=images,
            state_labels=class_indices.long(),
        )

        sample = transform_labels(sample, label_type=self.label_type, num_classes=self.meta_data.n_labels)

        self._current_iteration += 1
        if self._current_iteration >= self.meta_data.total_iterations:
            self.terminal = True

        return sample

    @profile
    def reset(self):
        self._iterator = self.data_generator()
        self.terminal = False
        self._current_iteration = 0
        self._total_iterations = self.meta_data.total_iterations
