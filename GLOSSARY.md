# Bike frame

A side-view model of a bicycle frame used to size its members under rider loads.

## Language

**Member**:
One structural line in the model. A pair of stays or fork legs is one member.
_Avoid_: Tube, element, beam

**Tube**:
The round cross-section of a member, given by outside diameter and wall.
_Avoid_: Section, profile

**Load point**:
Where the rider's weight is applied: the saddle, the bars at the stem clamp, or the pedals at the bottom bracket.
_Avoid_: Node, contact

**Load split**:
The share of the rider's weight at each load point. The three shares sum to the whole rider.
_Avoid_: Distribution, bias

**Load case**:
A named combination of rider mass, bump factor, and load split.
_Avoid_: Scenario, load step

**Bump factor**:
The total acceleration of a load case as a multiple of gravity. A bump factor of 1 is the rider sitting still.
_Avoid_: G-load, shock factor, dynamic factor

**Seat tube center-to-center**:
The distance along the seat tube from the bottom bracket to the top tube junction.
_Avoid_: Top tube drop, seat tube length

**Worst case**:
The load case that governs a member: the one with the lowest safety factor for that member.
_Avoid_: Envelope, max case

**Required wall**:
The wall thickness, at the member's current outside diameter, that meets the target safety factor.
_Avoid_: Gauge, minimum thickness
