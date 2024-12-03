import pandas as pd

from optilearn.utils.dataset.metadata.image import (
    make_image_training_metadata,
)

LABEL_COL = "Label"
FILENAME_COL = "FileName"


def get_class_distribution(data_path: str, labels: bool = True) -> pd.Series:
    meta_data = make_image_training_metadata(data_path)

    if labels:
        file_to_label = pd.DataFrame(
            meta_data.file_label_pairs, columns=["FileName", "Label"]
        )
        distribution = file_to_label["Label"].value_counts()
    else:
        file_to_class_index = pd.DataFrame(
            meta_data.file_class_index_pairs, columns=["FileName", "ClassIndex"]
        )
        distribution = file_to_class_index["ClassIndex"].value_counts()

    return distribution


def load_label_mapping(path: str) -> pd.DataFrame:
    label_mapping = pd.read_csv(path)

    assert len(label_mapping) >= 2

    label_mapping = label_mapping.iloc[:, :2]
    label_mapping.columns = [FILENAME_COL, LABEL_COL]

    # String extensions
    label_mapping.loc[:, FILENAME_COL] = (
        label_mapping[FILENAME_COL].str.split(".", expand=True)[0].astype("string")
    )

    return label_mapping


def non_critical_to_critical_list(
    data_path: str,
    non_critical_classes: list[str],
):
    all_classes = get_class_distribution(data_path).index.tolist()

    return list(set(all_classes) - set(non_critical_classes))


if __name__ == "__main__":
    non_critical_labels = (
        pd.read_excel(
            "/Users/sarosh/data/OptiLearn/ClassificationUsecase/data/Less_critical_Classes.xlsx",
            header=None,
        )
        .squeeze()
        .str.lower()
        .tolist()
    )

    critical = non_critical_to_critical_list(
        "/Users/sarosh/data/OptiLearn/ClassificationUsecase/data/converted/train",
        non_critical_classes=non_critical_labels,
    )

    pd.Series(critical).to_excel(
        "/Users/sarosh/data/OptiLearn/ClassificationUsecase/data/critical_classes.xlsx",
        index=False,
    )
