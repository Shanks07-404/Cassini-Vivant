# DEMO-FROST-PROTECTION: Illustrative Frost Protection Guidelines

> **⚠️ ILLUSTRATIVE DEMO GUIDELINES — not a real regulation.**
> This document is a fictional example created for demonstration purposes only.
> It does not prescribe mandatory legal or operational regulations.

**Document ID:** `DEMO-FROST-PROTECTION`  
**Issuer:** Cassini Vivant Decision Support System (Demo)  
**Version:** Demo v2  

---

## § 3.1 — Cassini Vivant Decision Guidance (Not an Appellation Rule)

In a Bayesian economic decision framework, the vineyard manager minimizes expected total cost by applying the cost-loss ratio rule:

**Activate protection on parcel _i_ if and only if the calibrated frost probability exceeds the action threshold:**

> p_frost(i) ≥ p* = C / L

where C is the per-parcel per-night protection cost and L is the expected crop loss if frost occurs unprotected.

*Note:* This rule is decision-support guidance to balance protection expenses against freeze damage; it is NOT an appellation rule or regulatory mandate. The action threshold p* is determined strictly by the economic parameters of the specific vineyard enterprise.

---

## § 3.2 — Limitations of Regional Forecasts During Inversion Nights

During radiative temperature inversion nights, regional weather stations measure above-zero temperatures at elevation while valley-floor parcels experience severe frost due to katabatic cold air pooling.

- A regional-alert strategy that triggers protection only when the regional forecast falls below zero will **completely miss** inversion frost events.
- Topographic features (relative elevation, slope, valley drainage axis) govern cold-pool accumulation.

---

## § 3.3 — Sensor Network Optimization

The base sensor network may be positioned predominantly on ridges and mid-slopes, leaving valley floors unmonitored. A decision-focused sensor placement algorithm prioritizes new stations in cold-pool-prone locations where uncertainty reduction most affects frost protection decisions.

Adding sensors in valley positions typically improves frost detection recall and reduces total expected cost per night by capturing the cold pooling signal directly. Technical modeling specifications and calibration factors are documented separately in `MODEL-CARD.md`.
