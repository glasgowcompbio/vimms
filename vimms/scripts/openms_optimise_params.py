from dataclasses import dataclass

@dataclass
class BaseParameters:
    MIN_RT: int = 0
    MAX_RT: int = 7200
    ISOLATION_WINDOW: float = 0.7
    DEFAULT_MS1_SCAN_WINDOW_START: float = 310.0
    DEFAULT_MS1_SCAN_WINDOW_END: float = 2000.0
    DEISOTOPE: bool = True
    CHARGE_RANGE_START: int = 2
    CHARGE_RANGE_END: int = 3
    MIN_FIT_SCORE: int = 80
    PENALTY_FACTOR: float = 1.5

    @classmethod
    def debug(cls):
        return cls(MAX_RT=1500)

@dataclass
class CommonParameters(BaseParameters):
    N: int = 15
    RT_TOL: int = 30
    MZ_TOL: int = 10
    MIN_MS1_INTENSITY: int = 5000

@dataclass
class TopNParameters(CommonParameters):
    EXCLUDE_AFTER_N_TIMES: int = 2
    EXCLUDE_T0: int = 15

@dataclass
class SmartROIParameters(CommonParameters):
    IIF: float = 10
    DP: float = 0.1
    MIN_ROI_INTENSITY: int = 500
    MIN_ROI_LENGTH: int = 0
    MIN_ROI_LENGTH_FOR_FRAGMENTATION: int = 0

@dataclass
class WeightedDEWParameters(CommonParameters):
    EXCLUDE_T0: int = 15

class ParametersBuilder:
    def __init__(self, parameters_class):
        self.parameters = parameters_class()

    def set(self, attribute, value):
        if hasattr(self.parameters, attribute):
            setattr(self.parameters, attribute, value)
        else:
            raise ValueError(f"{attribute} is not a valid attribute for {self.parameters.__class__.__name__}")
        return self

    def build(self):
        return self.parameters
