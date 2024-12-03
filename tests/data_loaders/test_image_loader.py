import os

import numpy as np

from optilearn.environments.data_loaders.learning.simple_image_loader import SimpleImageLoader


def test_image_loader_iterator(testing_data_dir):
    batch_size = 1
    n_test_images = 2

    iterations = int(np.ceil(batch_size / n_test_images))

    loader = SimpleImageLoader(data_path=os.path.join(testing_data_dir, "test_images"), transform=None, batch_size=1)

    batches = []
    for iterator in range(iterations):
        batches.append(loader.step())

    assert len(batches) == iterations
