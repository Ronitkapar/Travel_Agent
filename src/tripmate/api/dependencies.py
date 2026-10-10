"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from ..config import Settings, get_settings
from ..services.travel import TravelPlanner, get_planner


def planner_dependency() -> TravelPlanner:
    return get_planner()


def settings_dependency() -> Settings:
    return get_settings()


Planner = Annotated[TravelPlanner, Depends(planner_dependency)]
SettingsDep = Annotated[Settings, Depends(settings_dependency)]
