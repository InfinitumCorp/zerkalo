"""Zerkalo™ (Зеркало, "mirror") — reflexive simulation engine.

Pipeline: mirror the world → populate a synthetic public → rehearse a premise → read the forecast.
"""

from .config import Scenario, Premise, load_scenario
from .simulation import Simulation, RunResult

__all__ = ["Scenario", "Premise", "load_scenario", "Simulation", "RunResult"]
__version__ = "0.1.0"
