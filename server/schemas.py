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


class EngineSpec(BaseModel):
    name: str
    params: dict = {}


class PositionIn(BaseModel):
    pawns: list[list[int]]
    h_walls: list[list[int]] = []
    v_walls: list[list[int]] = []
    walls_left: list[int]
    turn: int


class AnalyzeRequest(BaseModel):
    position: PositionIn
    engines: list[EngineSpec] = []
