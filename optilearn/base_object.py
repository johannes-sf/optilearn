from dataclasses import dataclass
from typing import Any, ClassVar, List
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


@dataclass
class BaseDataClass:
    """
    Meant to be inherited by non-pydantic data classes.
    """

    @classmethod
    def get_field_names(cls) -> List[str]:
        """
        :return: List[str] List of all defined attributes of the data class
        """
        return list(cls.__annotations__.keys())


class BasePydanticModel(BaseModel):
    """
    Base model for all pydantic objects. Allows un-initialized classes
    as field values of pydantic models.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_field_names(cls) -> List[str]:
        """

        :return:
            List[str] List of all defined attributes of the data class
        """
        return list(cls.model_fields.keys())


class BaseObject(BasePydanticModel):
    """
    To be inherited for pydantic models where all fields of the model
    should have the same type, as defined in <abstract_base_type> class variable
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    abstract_base_type: ClassVar

    @model_validator(mode="after")
    def check_init_args(self) -> Self:
        for field in self.get_field_names():
            try:
                assert issubclass(getattr(self, field), self.abstract_base_type)
            except (TypeError, AssertionError):
                raise TypeError(
                    f"Field <{field}> of {self.__repr_name__()} is not of type <{self.abstract_base_type.__name__}>"
                )

        return self

    @classmethod
    @field_validator("*")
    def check_field_types(cls, init_args: dict) -> Self:
        for attribute, value in init_args:
            assert isinstance(attribute, cls.abstract_base_type), f"{attribute} not of type {cls.abstract_base_type}"

        return init_args
