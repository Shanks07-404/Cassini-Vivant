export interface ParcelProperties {
  id: string;
  name: string;
  centroid_lon: number;
  centroid_lat: number;
  elevation: number;
  rel_elev: number;
  slope_deg: number;
  aspect_sin: number;
  aspect_cos: number;
  is_valley_bottom: boolean;
  is_frost_pocket: boolean;
  area_ha: number;
  value: number;
  risk_score: number;
  is_simulated: boolean;
}

export interface ParcelFeature {
  type: 'Feature';
  id: string;
  geometry: {
    type: 'Polygon';
    coordinates: number[][][];
  };
  properties: ParcelProperties;
}

export interface ParcelGeoJSON {
  type: 'FeatureCollection';
  metadata: {
    count: number;
    crs: string;
    simulated: boolean;
    disclaimer: string;
  };
  features: ParcelFeature[];
}

export interface HoverInfo {
  x: number;
  y: number;
  parcel: ParcelProperties;
}

export interface Station {
  id: string;
  name: string;
  x: number;
  y: number;
  lon: number;
  lat: number;
  elevation: number;
  rel_elev: number;
  is_pocket: boolean;
}

export interface ParcelPrediction {
  parcel_id: string;
  name: string;
  elevation: number;
  rel_elev: number;
  mu: number;
  sigma: number;
  p_frost: number;
  is_critical: boolean;
  is_watch_tier: boolean;
  is_frost_pocket: boolean;
}

export interface PredictionResponse {
  night: string;
  hour: number;
  threshold_c: number;
  sensor_count: number;
  mean_uncertainty_sigma: number;
  predictions: ParcelPrediction[];
}

export interface NextSensorCandidate {
  parcel_id: string;
  name: string;
  lon: number;
  lat: number;
  rel_elev: number;
  current_sigma: number;
  decision_score: number;
}

export interface NextSensorRecommendation {
  best_parcel_id: string;
  best_parcel_name: string;
  best_coords: [number, number];
  rel_elev: number;
  elevation: number;
  current_sigma: number;
  expected_new_local_sigma: number;
  network_mean_sigma_current: number;
  network_mean_sigma_after: number;
  sigma_reduction: number;
  decision_score: number;
  ranked_candidates: NextSensorCandidate[];
}

export interface ReplayFrame {
  hour: number;
  time_label: string;
  regional_temp: number;
  sensor_readings: Record<string, number>;
  parcel_temperatures: Record<string, number>;
}

export interface ReplayData {
  night_id: string;
  profile: {
    t_min_regional: number;
    t_max_regional: number;
    is_hero_night: boolean;
    clear_calm_factor: number;
  };
  frames: ReplayFrame[];
}

export interface StrategyDetail {
  name: string;
  sensor_set?: string;
  alert_triggered?: boolean;
  parcels_protected: number | null;
  parcels_damaged: number | null;
  parcels_saved?: number | null;
  protection_cost: number | null;
  crop_loss: number | null;
  false_alarm_cost?: number | null;
  total_expenditure: number | null;
}

export interface TrainingNightBenchmark {
  mean: number;
  ci_95: [number, number];
}

export interface RoiResponse {
  label: string;
  night: string;
  is_hero_showcase?: boolean;
  regional_temp: number;
  parameters: {
    protect_threshold: number;
    cost_loss_threshold?: number;
    break_even_ratio?: number;
    cost_per_parcel_night: number;
    loss_per_parcel: number;
    total_parcels: number;
    truly_frozen_parcels: number;
    active_sensor_count: number;
    sensor_set_strategy_c?: string;
  };
  strategies: {
    strategy_a: StrategyDetail;
    strategy_b1?: StrategyDetail;
    strategy_b2?: StrategyDetail;
    strategy_c_k6?: StrategyDetail;
    strategy_c_k8?: StrategyDetail;
    strategy_b: StrategyDetail;
    strategy_c: StrategyDetail;
  };
  training_nights_benchmarks?: Record<string, TrainingNightBenchmark>;
  night_types_breakdown?: Record<string, Record<string, number>>;
  comparison: {
    break_even_ratio?: number;
    cost_loss_threshold?: number;
    net_benefit_vs_blanket: number;
    net_benefit_vs_nothing: number;
    net_benefit_vs_regional_alert?: number;
    protection_coverage_ratio: number;
    frost_strike_precision: number;
  };
}
