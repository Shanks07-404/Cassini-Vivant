# Cassini Vivant Model Card: Universal Kriging Microclimate Model

> **TECHNICAL DOCUMENTATION ONLY — NOT AN APPELLATION REGULATION**
> This model card documents the geostatistical architecture, hyperparameter calibration, and uncertainty quantification used by the Cassini Vivant decision support system.

---

## 1. Model Architecture

- **Framework:** Universal Kriging Gaussian Process regression with deterministic topographical trend.
- **Trend Specification:** $m(x, y, z_{\text{rel}}) = \beta_0 + \beta_1 \cdot z_{\text{rel}}$, where $z_{\text{rel}}$ is parcel centroid elevation above the local valley drainage axis in meters.
- **Covariance Kernel:** Stationary exponential covariance with spatial horizontal range $l_{xy}$ and vertical lapse scale $l_h$:
  $$k(x_i, x_j) = \sigma_f^2 \exp\left( - \sqrt{\frac{d_{xy}^2}{l_{xy}^2} + \frac{\Delta z_{\text{rel}}^2}{l_h^2}} \right) + \sigma_n^2 \delta_{ij}$$

---

## 2. Active Hyperparameters (Farm Calibration b)

- Horizontal spatial range $l_{xy}$: $527.2\text{ m}$
- Vertical lapse scale $l_h$: $148.9\text{ m}$
- Process variance $\sigma_f$: $0.585^\circ\text{C}$
- Measurement noise $\sigma_n$: $0.101^\circ\text{C}$
- Observation noise: Gaussian $\mathcal{N}(0, 0.30^2)$ applied to all physical weather stations.

---

## 3. Predictive Uncertainty Calibration

- **Empirical Calibration Scale Factor:** An empirical factor of $s = 1.855$ is applied to the raw kriging standard deviation $\sigma_{\text{geom}}$:
  $$\sigma_{\text{calibrated}} = s \cdot \sigma_{\text{geom}} = 1.855 \cdot \sigma_{\text{geom}}$$
- **Probability Formulation:** Calibrated frost probability is evaluated as:
  $$p_{\text{frost}} = \Phi\left( \frac{0.0 - \mu}{\sigma_{\text{calibrated}}} \right)$$
- **Out-of-Sample Validation:** Evaluated via 5-fold stratified cross-validation across training nights (002–060), ensuring nominal 90% prediction interval coverage (73.5% to 98.3% out-of-sample).
