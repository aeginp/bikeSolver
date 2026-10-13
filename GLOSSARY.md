# Bike frame

A side-view model of a bicycle frame used to size its members under rider loads.

## Model and loads

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

## Frame dimensions

**Seat tube**:
Center-to-center distance along the seat tube from the bottom bracket to the top tube junction.
_Avoid_: Seat tube center-to-center, center-to-top, top tube drop

**Top tube**:
Center-to-center length of the actual top tube, from the top tube junction to the head tube top.
_Avoid_: Effective top tube, horizontal top tube

**Effective top tube**:
Horizontal distance from the head tube top back to the seat tube axis. Always derived.
_Avoid_: Top tube, virtual top tube

**Axle-to-crown**:
Distance parallel to the steering axis from the bottom of the head tube to the front axle.
_Avoid_: Fork length

**Fork offset**:
Perpendicular distance from the steering axis to the front axle.
_Avoid_: Rake

## Results

**Worst case**:
The load case that governs a member: the one with the lowest safety factor for that member.
_Avoid_: Envelope, max case

**Required wall**:
The wall thickness, at the member's current outside diameter, that meets the target safety factor.
_Avoid_: Gauge, minimum thickness
