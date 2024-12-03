import os

import pytest


@pytest.fixture
def testing_data_dir():
    return os.path.join(os.path.dirname(__file__), "testing_data")
