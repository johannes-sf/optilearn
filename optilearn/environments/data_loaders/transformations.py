class AbstractTransformationPipeline:
    """
    A pipeline of transformations to be applied to data.
    """

    is_fitted: bool = False

    def __init__(self, *args, **kwargs):
        self.pipeline = None
        self._feature_names = []
        self._input_shape = ()
        self._output_shape = ()

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        pass

    def transform(self, data):
        self._feature_names = data.columns
        return data

    def fit_transform(self, data):
        return data

    def __call__(self, data):
        return data

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
