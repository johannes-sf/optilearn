import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from optilearn.configs.constants import LabelType
from optilearn.environments.data_loaders.abstract_data_loader import (
    AbstractDataLoader,
)
from optilearn.environments.responses import LearningImageLoaderResponse
from optilearn.utils.dataset.metadata.dto import TrainingMetaData
from optilearn.utils.dataset.metadata.image import (
    make_image_training_metadata,
)
from optilearn.utils.dataset.label_transformations import (
    transform_labels,
)
from optilearn.utils.utils import insert_in_tensor


# ToDo: Simplify class, remove obsolete attributes/methods


class LearningImageLoader(AbstractDataLoader):
    random_generator = np.random.RandomState(seed=0)
    # ToDo: Improve augmentation
    # ToDo: Get transform values from config

    augmentation_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.1306], std=[0.3081]),
        ]
    )

    def __init__(
        self,
        data_path: str,
        learning_mode: bool = True,
        shuffle: bool = True,
        batch_size: int = None,
        contamination: float = 0.0,
        min_mixture_ratio: float = 0.2,
        class_subset: list = None,
        label_type: LabelType = LabelType.NORMAL,
        device: str = "cpu",
        img_size: tuple = (255, 255),
        **kwargs,
    ):
        super(LearningImageLoader, self).__init__(
            learning_mode=learning_mode,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )

        self.data_path = data_path
        self.contamination = contamination
        self.min_mixture_ratio = min_mixture_ratio
        self.class_subset = class_subset
        self.label_type = label_type
        self.terminal = False
        self.device = device

        self.meta_data: TrainingMetaData = make_image_training_metadata(
            self.data_path,
            shuffle=shuffle,
            batch_size=batch_size,
            class_subset=class_subset,
        )
        self._total_iterations: int = self.meta_data.total_iterations

        self.image_size = img_size
        self.norm_mean = kwargs.get("norm_means", [0.5, 0.5, 0.5])
        self.norm_std = kwargs.get("norm_std", [0.5, 0.5, 0.5])
        self.rotation = kwargs.get("rotation", 30)

        self.resize: Callable = transforms.Resize(self.image_size)
        self.to_tensor: Callable = transforms.ToTensor()
        if learning_mode:
            self.transform_pipeline: Callable = torch.nn.Sequential(
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(self.rotation),
            )
        else:
            self.transform_pipeline: Callable = torch.nn.Sequential(
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            )
        self.augmentation_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                self.transform_pipeline,
            ]
        )

    def step(self) -> LearningImageLoaderResponse:
        images = torch.tensor([])
        class_indices = torch.tensor([], dtype=torch.int8)
        contaminating_indices = torch.tensor([], dtype=torch.int8)
        ratios = torch.tensor([], dtype=torch.float16)

        # ToDo: Profile Timing & Consider parallelization
        for element in range(self.batch_size):
            try:
                image_index: int = next(self.meta_data.index_iterator)
                image, class_index = self.sample_at_index(image_index, augment=False)

                if self.learning_mode:
                    if self.contamination:
                        image, contaminant_class, ratio = self.contaminate_image(
                            image, class_index, min_mixture_ratio=self.min_mixture_ratio
                        )

                        ratios = insert_in_tensor(ratios, ratio)
                        contaminating_indices = insert_in_tensor(contaminating_indices, contaminant_class)

                    image = self.augmentation_transforms(image)

                images = torch.cat([images, image.unsqueeze(0)])
                class_indices = insert_in_tensor(class_indices, class_index)

            except StopIteration:
                self.terminal = True
                break

        self._current_iteration += 1

        if (self.current_iteration + 1) == self.total_iterations:
            self.terminal = True

        images = images.to(self.device)
        class_indices = class_indices.to(self.device)
        contaminating_indices = contaminating_indices.to(self.device)
        ratios = ratios.to(self.device)

        sample = LearningImageLoaderResponse(
            state=images,
            state_labels=class_indices.long(),
            mixture_ratios=ratios.float(),
            contaminant_indices=contaminating_indices.long() if len(contaminating_indices) else None,
        )

        sample = transform_labels(
            sample,
            label_type=self.label_type,
            num_classes=self.meta_data.label_to_class_index
        )

        return sample

    @classmethod
    def sample_mixture_ratio(cls, min_ratio=0.6) -> float:
        return cls.random_generator.uniform(low=min_ratio, high=1)

    def contaminate_image(self, image: Tensor, class_index: int, min_mixture_ratio=0.6) -> tuple:
        if self.random_generator.binomial(1, self.contamination):
            # Sample randomly from all labels except the current
            index_subset = list(set(self.meta_data.class_index_to_label.keys()) - {class_index})

            contaminant, contaminant_index = self.sample_randomly(augment=False, class_index_subset=index_subset)
            image, ratio = self._contaminate(image, contaminant, min_mixture_ratio)

            return image, contaminant_index, ratio
        return image, class_index, 1

    def _contaminate(self, image, contaminant, min_mixture_ratio):
        mixture_ratio = self.sample_mixture_ratio(min_ratio=min_mixture_ratio)
        image = (mixture_ratio * image) + ((1 - mixture_ratio) * contaminant)

        return image, mixture_ratio

    def load_image(self, file_path: str, augment=False) -> Tensor:
        image: Tensor = self.to_tensor(default_loader(file_path))

        if augment:
            image = self.augmentation_transforms(image)

        image = self.resize(image)

        return image

    def sample_at_index(self, image_index, augment=False):
        image_path, class_index = self.meta_data.file_class_index_pairs[image_index]

        return self.load_image(image_path, augment), class_index

    def sample_randomly(self, augment: bool = False, class_index_subset: list = None):
        path_to_class_index = self.meta_data.file_class_index_pairs

        if class_index_subset is not None:
            path_to_class_index = list(
                filter(lambda x: x[1] in class_index_subset, path_to_class_index),
            )

        image_path, class_index = random.choice(path_to_class_index)

        return self.load_image(image_path, augment), class_index

    def sample_from_class(self, class_label, augment=False):
        if not self.learning_mode:
            raise ValueError(f"ImageLoader is not in Training mode.")

        class_images = self.meta_data.labels_to_files[class_label]

        return self.load_image(random.sample(class_images, 1)[0], augment), class_label

    def reset(self):
        self.terminal = False
        self.meta_data = make_image_training_metadata(
            self.data_path,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            class_subset=self.class_subset,
        )
        self._current_iteration = -1
        self._total_iterations = self.meta_data.total_iterations

    @property
    def total_iterations(self) -> int:
        return self.meta_data.total_iterations


if __name__ == "__main__":
    loader = LearningImageLoader(
        data_path="/Users/sarosh/data/OptiLearn/MNIST/train",
        batch_size=1,
        shuffle=True,
        contamination=1,
    )
    _ = loader.step()

    # cls_img, cls_label = loader.sample_from_class(9, augment=True)
    # rnd_img, rnd_label = loader.sample_randomly(augment=True)
    # ind_img, ind_label = loader.sample_at_index(22, augment=True)
