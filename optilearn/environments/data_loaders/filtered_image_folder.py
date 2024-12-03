from torchvision.datasets import ImageFolder

from optilearn.utils.utils import get_device


class FilteredImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform=None,
        class_subset: list = None,
        device=get_device(),
        target_transform=None,
        is_valid_file=None,
    ):
        ImageFolder.__init__(
            self,
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        if class_subset is not None:
            subset_class_to_idx = {label: index for label, index in self.class_to_idx.items() if label in class_subset}
            re_indexed_class_to_idx = {
                label: iterator for iterator, (label, _) in enumerate(subset_class_to_idx.items())
            }

            self.samples = [sample for sample in self.samples if str(sample[1]) in class_subset]
            self.targets = [s[1] for s in self.samples]
            self.classes = class_subset
            self.class_to_idx = re_indexed_class_to_idx
