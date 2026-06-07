from typing import Union
from pydantic import BaseModel


class ControllerObj(BaseModel):
    name: str
    params: dict = {}


class NewGame(BaseModel):
    controllers: list[Union[str, ControllerObj]]


class MoveIn(BaseModel):
    type: str
    to: list[int] | None = None
    c: int | None = None
    r: int | None = None
    orient: str | None = None
