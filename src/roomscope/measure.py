"""The one numeric type in the output contract: a value with a calibrated interval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    value: float
    lo: float
    hi: float
    unit: str = "m"
    level: float = 0.9
    method: str = "uncalibrated"

    @classmethod
    def from_rel(cls, value: float, rel_halfwidth: float, **kw) -> "Measurement":
        """Symmetric relative interval, e.g. rel_halfwidth=0.08 -> value +/- 8%."""
        d = abs(value) * rel_halfwidth
        return cls(value, value - d, value + d, **kw)

    @classmethod
    def from_abs(cls, value: float, halfwidth: float, **kw) -> "Measurement":
        return cls(value, value - halfwidth, value + halfwidth, **kw)

    def to_json(self) -> dict:
        return {
            "value": round(self.value, 4),
            "lo": round(self.lo, 4),
            "hi": round(self.hi, 4),
            "unit": self.unit,
            "level": self.level,
            "method": self.method,
        }
