import os
from io import StringIO
from typing import Iterable

import pandas as pd

import optilearn
from optilearn.inference.inferencer import AbstractInferencer


class SKInferencer(AbstractInferencer):
    def __int__(self, model_version: str, device: str = "cpu", local_registry: str = ""):
        super().__init__(model_version, device, local_registry)

    def infer(self, model_input: Iterable, preference: list[float] = None, **kwargs) -> dict[str:float]:
        probabilities = self.model.model.predict_proba(self.process_input(model_input))

        return {self.class_index_to_label[index]: probability for index, probability in enumerate(probabilities)}

    def process_input(self, model_input: Iterable) -> Iterable:
        return self.transformation_pipeline.transform(model_input)

    def process_preference(self, preference: list[float] = None) -> Iterable:
        pass


if __name__ == "__main__":
    import vaex

    LOCAL_REGISTRY = os.path.join(os.path.dirname(os.path.dirname(optilearn.__file__)), "model_registry")

    text = """PetId,StateProvince,AverageDeductible,TotalWrittenPremium,EarnedPremium,WorkingDog,TotalClaims,PaidClaims,TotalPaidAmount,TotalClaimedAmount,DurationInDays
   can398144101zoe,CAN-AB,300.0,1478.82,1385.218033,False,1.0,0.0,0.0,167.6,1095"""

    df = vaex.from_pandas(pd.read_csv(StringIO(text)).set_index("PetId"))

    # df = pd.read_csv(StringIO(text)).set_index("PetId")

    inferencer = SKInferencer(model_version="OP-TABCLALGB-42", local_registry=LOCAL_REGISTRY)

    inferencer.infer(df)
