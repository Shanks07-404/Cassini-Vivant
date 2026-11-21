# Glossary: Frost Protection & Microclimate Terms

> **⚠️ ILLUSTRATIVE DEMO RULE — not a real regulation.**
> General-purpose definitions for demonstration only.

---

## Temperature Inversion

A meteorological condition where air temperature increases with altitude (opposite of the normal lapse rate). During clear, calm nights, the ground radiates heat to space, cooling the surface layer. The warmer air aloft acts as a lid, trapping cold air near the surface.

## Katabatic Flow (Cold Air Drainage)

Gravity-driven downslope flow of dense, cold air from elevated terrain into valleys, hollows, and topographic depressions. This mechanism concentrates frost risk in valley floors.

## Cold Pool / Frost Pocket

A topographic depression where cold air accumulates due to katabatic drainage and stagnates because surrounding terrain prevents outflow. Temperatures in frost pockets can be several degrees colder than nearby slopes.

## Cost-Loss Ratio

The ratio C/L where C is the cost of protection per parcel per night and L is the expected crop loss if frost occurs unprotected. The optimal action threshold under expected cost minimization is p* = C/L.

## Universal Kriging (GP with Trend)

A geostatistical interpolation method that models spatial correlation of temperature residuals around a trend function (here: linear in relative elevation). Produces both a best estimate (mean) and an uncertainty estimate (standard deviation) at unobserved locations.

## Relative Elevation

The elevation of a parcel centroid relative to the local valley floor, computed from the DEM (Digital Elevation Model). Parcels with low relative elevation are more prone to cold air pooling.

## Radiative Frost

Frost caused by radiative cooling of the ground surface on clear, calm nights — distinct from advective frost caused by cold air masses. Radiative frost is associated with temperature inversions.

## Phenological Stage

The developmental stage of the vine (bud dormancy → bud swell → green tip → leaf out). Frost sensitivity increases sharply once buds break dormancy.

## Brier Score

A scoring rule for probabilistic predictions, measuring the mean squared difference between predicted probabilities and observed binary outcomes (frost / no frost). Lower is better.

## Reliability Diagram

A calibration plot showing observed frost frequency (y-axis) versus predicted frost probability (x-axis) in bins. A perfectly calibrated model falls on the diagonal.
