"""Load cases. Fractions are shares of rider weight and sum to 1."""

from __future__ import annotations

from dataclasses import dataclass

# Standard gravity. The bump factor already includes gravity, so 1 is sitting still.
GRAVITY = 9.81


@dataclass(frozen=True)
class LoadCase:
    name: str
    mass_kg: float
    bump_g: float
    saddle: float
    bars: float
    pedals: float

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Every load case needs a name.")
        if self.mass_kg <= 0:
            raise ValueError(f"{self.name}: rider mass must be positive.")
        if self.bump_g <= 0:
            raise ValueError(f"{self.name}: bump factor must be positive.")
        for label, share in (("saddle", self.saddle), ("bars", self.bars), ("pedals", self.pedals)):
            if share < 0:
                raise ValueError(f"{self.name}: {label} share cannot be negative.")
        total = self.saddle + self.bars + self.pedals
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{self.name}: load split sums to {total * 100:.1f}%, not 100%.")

    @property
    def total_force(self) -> float:
        """Downward rider force in newtons."""
        return self.mass_kg * GRAVITY * self.bump_g

    def forces(self) -> dict[str, float]:
        """Downward force at each load point, in newtons."""
        total = self.total_force
        return {
            "saddle": total * self.saddle,
            "stem_clamp": total * self.bars,
            "bb": total * self.pedals,
        }
