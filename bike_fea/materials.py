"""Frame materials. Modulus and strength are MPa."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    youngs_modulus: float
    yield_strength: float
    allowable_compression: float | None = None

    def compression_limit(self) -> float:
        """Strength used when the governing stress is compressive."""
        if self.allowable_compression is None:
            return self.yield_strength
        return self.allowable_compression


# E and yield are typical tube values, not a certification database.
BUILTINS: dict[str, Material] = {
    "4130": Material("4130", 205_000.0, 460.0),
    "6061-T6": Material("6061-T6", 68_900.0, 276.0),
    "3Al-2.5V": Material("3Al-2.5V", 105_000.0, 620.0),
}
