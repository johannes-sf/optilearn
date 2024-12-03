import logging
from typing import Callable

import matplotlib.pyplot as plt
from torch import Tensor, stack
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from optilearn.configs.constants import IMAGE_FILE_EXTENSIONS
from optilearn.environments.data_loaders.abstract_data_loader import AbstractDataLoader
from optilearn.environments.responses import DataLoaderResponse
from optilearn.utils.utils import list_all_files


class InferenceImageLoader(AbstractDataLoader):
    def __init__(self, data_path: str, shuffle: bool = False, batch_size: int = None, **kwargs):
        if "learning_mode" in kwargs:
            kwargs.pop("learning_mode")
        super(InferenceImageLoader, self).__init__(
            learning_mode=False,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )
        self.data_path = data_path

        self._iterator = self._make_iterator()

        self.terminal = False

        # ToDo: Get transform values from config
        self.image_dim = (28, 28)
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        self.transform_pipeline: Callable = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

    def step(self) -> DataLoaderResponse:
        images = []

        if self.terminal:
            logging.warning("Image iterator exhausted. Reset environment.")
            return DataLoaderResponse(None, None)

        for element in range(self.batch_size):
            try:
                images.append(next(self.iterator))

            except StopIteration:
                self.terminal = True
                break

        self._current_iteration += 1

        if self._current_iteration >= self._total_iterations:
            self.terminal = True

        return DataLoaderResponse(state=stack(images), state_labels=None)

    def _load_image(self, file_path: str) -> Tensor:
        # ToDo: Clean-up
        _plot = False
        image = self.transform_pipeline(default_loader(file_path))

        if _plot:
            self.__plot__(image)

        return image

    @staticmethod
    def __plot__(image):
        plt.imshow(image.permute(1, 2, 0))

    def reset(self):
        self.terminal = False
        self._iterator = self._make_iterator()
        self._current_iteration = 0

    def _make_iterator(self):
        image_paths = list_all_files(path=self.data_path, extensions=IMAGE_FILE_EXTENSIONS)

        self._total_iterations = len(image_paths)

        def image_iterator(files):
            for image_file in files:
                yield self.transform_pipeline(default_loader(image_file))

        return image_iterator(image_paths)


if __name__ == "__main__":
    loader = InferenceImageLoader(
        data_path="/Users/sarosh/data/OptiLearn/MNIST/infer/",
        batch_size=1,
    )
    b = []
    for _ in range(100):
        b.append(loader.step())
