import { useEffect, useRef, useState } from 'react';
import maplibregl, { Map as MapLibreMap, Marker } from 'maplibre-gl';
import { ParcelFeature, ParcelGeoJSON, ParcelProperties, HoverInfo, Station, ParcelPrediction, NextSensorRecommendation } from '../types';
import { getTranslation, Language } from '../i18n';

interface CassiniMapProps {
  lang: Language;
  selectedParcel: ParcelProperties | null;
  onSelectParcel: (parcel: ParcelProperties | null) => void;
  stations: Station[];
  predictions: Record<string, ParcelPrediction>;
  meshGeoJSON: any | null;
  showMesh: boolean;
  onToggleMesh: () => void;
  showUncertainty: boolean;
  onToggleUncertainty: () => void;
  showTruthOverlay: boolean;
  onToggleTruthOverlay: () => void;
  truthFrozenParcelIds: Set<string>;
  showFrostPocketsDebug: boolean;
  onToggleFrostPocketsDebug: () => void;
  recommendation: NextSensorRecommendation | null;
  onAddSensorAtParcel: (parcelId: string, lat: number, lon: number, name: string) => void;
}

export default function CassiniMap({
  lang,
  selectedParcel,
  onSelectParcel,
  stations,
  predictions,
  meshGeoJSON,
  showMesh,
  onToggleMesh,
  showUncertainty,
  onToggleUncertainty,
  showTruthOverlay,
  onToggleTruthOverlay,
  truthFrozenParcelIds,
  showFrostPocketsDebug,
  onToggleFrostPocketsDebug,
  recommendation,
  onAddSensorAtParcel,
}: CassiniMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [parcelsData, setParcelsData] = useState<ParcelGeoJSON | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const stationMarkersRef = useRef<Marker[]>([]);
  const beaconMarkerRef = useRef<Marker | null>(null);

  const t = getTranslation(lang);

  // 1. Fetch GeoJSON from backend
  useEffect(() => {
    fetch('/api/parcels')
      .then((res) => res.json())
      .then((data: ParcelGeoJSON) => {
        setParcelsData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load parcels:', err);
        setLoading(false);
      });
  }, []);

  // 2. Initialize offline MapLibre instance
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          {
            id: 'parchment-background',
            type: 'background',
            paint: {
              'background-color': '#f5efe0', // Antique parchment tone
            },
          },
        ],
      },
      center: [3.8000, 47.8100],
      zoom: 13.4,
      minZoom: 11,
      maxZoom: 16,
      attributionControl: false,
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 3. Setup Layers once Map and parcelsData are ready
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !parcelsData) return;

    const setupLayers = () => {
      // Inject live predictions into feature properties if available
      const enrichedData = {
        ...parcelsData,
        features: parcelsData.features.map((feat) => {
          const pred = predictions[feat.properties.id];
          const isFrozenTruth = truthFrozenParcelIds.has(feat.properties.id);
          return {
            ...feat,
            properties: {
              ...feat.properties,
              p_frost: pred ? pred.p_frost : feat.properties.value,
              mu: pred ? pred.mu : 0,
              sigma: pred ? pred.sigma : 0.75,
              is_truly_frozen: isFrozenTruth,
              high_uncertainty: pred ? pred.sigma >= 0.80 : false,
            },
          };
        }),
      };

      if (map.getSource('parcels')) {
        (map.getSource('parcels') as maplibregl.GeoJSONSource).setData(enrichedData as any);
      } else {
        map.addSource('parcels', {
          type: 'geojson',
          data: enrichedData as any,
        });

        // 3a. Parcel Fills (colored green -> amber -> red by p_frost)
        map.addLayer({
          id: 'parcels-fill',
          type: 'fill',
          source: 'parcels',
          paint: {
            'fill-color': [
              'interpolate',
              ['linear'],
              ['get', 'p_frost'],
              0.0, '#4d7c0f',   // Safe green
              0.2, '#84cc16',  // Lime green (alert threshold)
              0.5, '#eab308',   // Warning amber (critical)
              0.8, '#ea580c',   // Orange
              1.0, '#b91c1c',   // Severe frost critical red
            ],
            'fill-opacity': 0.78,
          },
        });

        // 3b. Uncertainty hatching / high-uncertainty layer
        map.addLayer({
          id: 'parcels-uncertainty',
          type: 'line',
          source: 'parcels',
          filter: ['==', ['get', 'high_uncertainty'], true],
          layout: {
            visibility: showUncertainty ? 'visible' : 'none',
          },
          paint: {
            'line-color': '#8c6d46',
            'line-width': 1.8,
            'line-dasharray': [3, 2],
          },
        });

        // 3c. Truth overlay (simulation ground truth T < 0°C)
        map.addLayer({
          id: 'parcels-truth-overlay',
          type: 'line',
          source: 'parcels',
          filter: ['==', ['get', 'is_truly_frozen'], true],
          layout: {
            visibility: showTruthOverlay ? 'visible' : 'none',
          },
          paint: {
            'line-color': '#0284c7', // Ice blue
            'line-width': 3.2,
          },
        });

        // 3d. Fine antique sepia parcel boundaries
        map.addLayer({
          id: 'parcels-borders',
          type: 'line',
          source: 'parcels',
          paint: {
            'line-color': '#8c6d46',
            'line-width': 1.1,
            'line-opacity': 0.85,
          },
        });

        // 3e. Hover highlight outline
        map.addLayer({
          id: 'parcels-hover-outline',
          type: 'line',
          source: 'parcels',
          filter: ['==', ['get', 'id'], ''],
          paint: {
            'line-color': '#2c2113',
            'line-width': 2.4,
          },
        });

        // 3f. Selected parcel outline
        map.addLayer({
          id: 'parcels-selected-outline',
          type: 'line',
          source: 'parcels',
          filter: ['==', ['get', 'id'], ''],
          paint: {
            'line-color': '#1e3a8a',
            'line-width': 3.2,
          },
        });

        // 3g. Debug frost-pockets layer (dashed blue highlight on low-lying valley floor)
        map.addLayer({
          id: 'parcels-frost-pockets',
          type: 'line',
          source: 'parcels',
          filter: ['==', ['get', 'is_frost_pocket'], true],
          layout: {
            visibility: showFrostPocketsDebug ? 'visible' : 'none',
          },
          paint: {
            'line-color': '#0284c7',
            'line-width': 2.5,
            'line-dasharray': [2, 2],
          },
        });

        // Mouse interactions
        map.on('mousemove', 'parcels-fill', (e) => {
          if (!e.features || e.features.length === 0) return;
          map.getCanvas().style.cursor = 'pointer';
          const feat = e.features[0] as unknown as ParcelFeature;
          const props = feat.properties;

          map.setFilter('parcels-hover-outline', ['==', ['get', 'id'], props.id]);

          setHoverInfo({
            x: e.point.x,
            y: e.point.y,
            parcel: props,
          });
        });

        map.on('mouseleave', 'parcels-fill', () => {
          map.getCanvas().style.cursor = '';
          map.setFilter('parcels-hover-outline', ['==', ['get', 'id'], '']);
          setHoverInfo(null);
        });

        map.on('click', 'parcels-fill', (e) => {
          if (!e.features || e.features.length === 0) return;
          const feat = e.features[0] as unknown as ParcelFeature;
          onSelectParcel(feat.properties);
        });
      }
    };

    if (map.isStyleLoaded()) {
      setupLayers();
    } else {
      map.once('load', setupLayers);
    }
  }, [parcelsData, predictions, truthFrozenParcelIds]);

  // 4. Update Delaunay Triangles Mesh Layer
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    if (!meshGeoJSON) {
      if (map.getLayer('cassini-mesh-lines')) map.removeLayer('cassini-mesh-lines');
      if (map.getSource('cassini-mesh')) map.removeSource('cassini-mesh');
      return;
    }

    if (map.getSource('cassini-mesh')) {
      (map.getSource('cassini-mesh') as maplibregl.GeoJSONSource).setData(meshGeoJSON);
    } else {
      map.addSource('cassini-mesh', {
        type: 'geojson',
        data: meshGeoJSON,
      });

      map.addLayer({
        id: 'cassini-mesh-lines',
        type: 'line',
        source: 'cassini-mesh',
        layout: {
          visibility: showMesh ? 'visible' : 'none',
        },
        paint: {
          'line-color': '#b24c28', // Cassini rust-orange
          'line-width': 1.8,
          'line-dasharray': [4, 3],
          'line-opacity': 0.85,
        },
      });
    }

    if (map.getLayer('cassini-mesh-lines')) {
      map.setLayoutProperty('cassini-mesh-lines', 'visibility', showMesh ? 'visible' : 'none');
    }
  }, [meshGeoJSON, showMesh]);

  // 5. Update toggles (Uncertainty, Truth overlay, Frost pocket debug)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    if (map.getLayer('parcels-uncertainty')) {
      map.setLayoutProperty('parcels-uncertainty', 'visibility', showUncertainty ? 'visible' : 'none');
    }
    if (map.getLayer('parcels-truth-overlay')) {
      map.setLayoutProperty('parcels-truth-overlay', 'visibility', showTruthOverlay ? 'visible' : 'none');
    }
    if (map.getLayer('parcels-frost-pockets')) {
      map.setLayoutProperty('parcels-frost-pockets', 'visibility', showFrostPocketsDebug ? 'visible' : 'none');
    }
  }, [showUncertainty, showTruthOverlay, showFrostPocketsDebug]);

  // 6. Selected Parcel Outline filter
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded() || !map.getLayer('parcels-selected-outline')) return;

    if (selectedParcel) {
      map.setFilter('parcels-selected-outline', ['==', ['get', 'id'], selectedParcel.id]);
    } else {
      map.setFilter('parcels-selected-outline', ['==', ['get', 'id'], '']);
    }
  }, [selectedParcel]);

  // 7. Render Station Markers (Bell Tower icon)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Clear existing markers
    stationMarkersRef.current.forEach((m) => m.remove());
    stationMarkersRef.current = [];

    stations.forEach((st) => {
      const el = document.createElement('div');
      el.className = 'station-bell-tower-marker group cursor-pointer relative';
      el.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-[#2c2113] border-2 border-[#ebdcc0] text-[#faf7f0] flex items-center justify-center shadow-md transform transition-transform group-hover:scale-125">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/>
            <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>
          </svg>
        </div>
        <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block bg-[#2c2113] text-[#faf7f0] px-2 py-1 rounded text-[10px] font-mono whitespace-nowrap shadow-lg z-50">
          ${st.id}: ${st.name} (${st.elevation.toFixed(0)}m)
        </div>
      `;

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([st.lon, st.lat])
        .addTo(map);

      stationMarkersRef.current.push(marker);
    });
  }, [stations]);

  // 8. Render Pulsing Beacon Marker for #1 Next Sensor Recommendation
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (beaconMarkerRef.current) {
      beaconMarkerRef.current.remove();
      beaconMarkerRef.current = null;
    }

    if (recommendation) {
      const [lon, lat] = recommendation.best_coords;
      const el = document.createElement('div');
      el.className = 'next-sensor-beacon cursor-pointer relative group';
      el.innerHTML = `
        <div class="relative flex items-center justify-center w-8 h-8">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#b24c28] opacity-75"></span>
          <div class="relative inline-flex rounded-full h-5 w-5 bg-[#b24c28] border-2 border-[#faf7f0] text-[#faf7f0] items-center justify-center shadow-lg font-mono font-bold text-[10px]">
            +
          </div>
        </div>
        <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-[#b24c28] text-[#faf7f0] px-2.5 py-1 rounded text-[10px] font-sans font-bold whitespace-nowrap shadow-xl z-50">
          Optimal Site: ${recommendation.best_parcel_id}
        </div>
      `;

      el.addEventListener('click', () => {
        onAddSensorAtParcel(
          recommendation.best_parcel_id,
          lat,
          lon,
          recommendation.best_parcel_name
        );
      });

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([lon, lat])
        .addTo(map);

      beaconMarkerRef.current = marker;
    }
  }, [recommendation]);

  const activeHoverPrediction = hoverInfo ? predictions[hoverInfo.parcel.id] : null;

  return (
    <div className="relative w-full h-full overflow-hidden bg-[#f5efe0]">
      {/* MapLibre DOM Node */}
      <div ref={mapContainer} className="w-full h-full" />

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#faf7f0]/80 z-20">
          <div className="text-center font-serif text-[#4a3821]">
            <div className="text-base font-bold animate-pulse">
              Chargement du cadastre de Cassini...
            </div>
            <div className="text-xs text-[#8c6d46]">Triangulation en cours...</div>
          </div>
        </div>
      )}

      {/* HTML Hover Tooltip */}
      {hoverInfo && (
        <div
          className="absolute z-30 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 px-3.5 py-2.5 rounded-lg shadow-xl border border-[#8c6d46]/50 bg-[#faf7f0]/95 backdrop-blur-sm text-xs font-serif text-[#2c2113] min-w-[210px]"
          style={{
            left: `${hoverInfo.x}px`,
            top: `${hoverInfo.y}px`,
          }}
        >
          <div className="flex items-center justify-between gap-3 border-b border-[#ebdcc0] pb-1 mb-1.5 font-bold">
            <span className="font-mono text-xs text-[#8c6d46]">{hoverInfo.parcel.id}</span>
            <span className="truncate">{hoverInfo.parcel.name}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] text-[#4a3821] font-sans">
            <span>{t.parcel.elevation}:</span>
            <span className="font-mono text-right font-medium">{hoverInfo.parcel.elevation} m</span>
            <span>{t.parcel.rel_elevation}:</span>
            <span className="font-mono text-right font-medium">+{hoverInfo.parcel.rel_elev} m</span>
            
            {activeHoverPrediction && (
              <>
                <span>{t.parcel.predicted_temp}:</span>
                <span className={`font-mono text-right font-bold ${activeHoverPrediction.mu < 0 ? 'text-[#b91c1c]' : 'text-[#4d7c0f]'}`}>
                  {activeHoverPrediction.mu > 0 ? `+${activeHoverPrediction.mu.toFixed(1)}` : activeHoverPrediction.mu.toFixed(1)}°C
                </span>
                <span>{t.parcel.uncertainty}:</span>
                <span className="font-mono text-right font-medium">±{activeHoverPrediction.sigma.toFixed(2)}°C</span>
                <span>{t.parcel.frost_prob}:</span>
                <span className={`font-mono text-right font-bold text-xs ${activeHoverPrediction.p_frost >= 0.5 ? 'text-[#b91c1c]' : activeHoverPrediction.p_frost >= 0.2 ? 'text-[#ea580c]' : 'text-[#4d7c0f]'}`}>
                  {(activeHoverPrediction.p_frost * 100).toFixed(0)}%
                </span>
              </>
            )}
          </div>

          {hoverInfo.parcel.is_frost_pocket && (
            <div className="mt-1.5 pt-1 border-t border-[#ebdcc0] text-[10px] font-sans font-semibold text-[#0284c7] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#0284c7]"></span>
              {t.parcel.frost_pocket_badge}
            </div>
          )}

          {truthFrozenParcelIds.has(hoverInfo.parcel.id) && showTruthOverlay && (
            <div className="mt-1 text-[10px] font-sans font-bold text-[#b91c1c] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#b91c1c]"></span>
              True Temperature &lt; 0°C (Frozen)
            </div>
          )}
        </div>
      )}

      {/* Map Control Layers & Legend Panel */}
      <div className="absolute bottom-5 left-5 z-10 p-3.5 rounded-lg border border-[#8c6d46]/40 bg-[#faf7f0]/95 backdrop-blur-sm shadow-lg font-serif text-xs text-[#38281a] max-w-xs space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="font-bold text-[#2c2113] text-xs tracking-wide">
            {t.map.legend_frost_prob}
          </span>
          <span className="px-1.5 py-0.5 text-[9px] font-sans font-bold uppercase rounded bg-[#4d7c0f]/15 text-[#4d7c0f] border border-[#4d7c0f]/30">
            {t.map.placeholder_badge}
          </span>
        </div>

        {/* Gradient Bar */}
        <div className="h-2.5 w-48 rounded overflow-hidden border border-[#8c6d46]/30 bg-gradient-to-r from-[#4d7c0f] via-[#eab308] to-[#b91c1c] shadow-inner" />

        <div className="flex justify-between text-[10px] text-[#8c6d46] font-sans font-medium">
          <span>{t.map.low_risk}</span>
          <span>{t.map.med_risk}</span>
          <span>{t.map.high_risk}</span>
        </div>

        {/* Layer Toggles */}
        <div className="pt-2 border-t border-[#ebdcc0] space-y-1.5 font-sans text-[11px]">
          {/* Cassini Triangles (Delaunay) Toggle */}
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="mesh-toggle" className="text-[#4a3821] cursor-pointer select-none">
              {t.map.mesh_toggle}
            </label>
            <input
              id="mesh-toggle"
              type="checkbox"
              checked={showMesh}
              onChange={onToggleMesh}
              className="h-3.5 w-3.5 rounded border-[#8c6d46] text-[#b24c28] focus:ring-0 cursor-pointer"
            />
          </div>

          {/* Model Uncertainty Toggle (Neutral label) */}
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="uncertainty-toggle" className="text-[#4a3821] cursor-pointer select-none">
              {t.map.uncertainty_toggle}
            </label>
            <input
              id="uncertainty-toggle"
              type="checkbox"
              checked={showUncertainty}
              onChange={onToggleUncertainty}
              className="h-3.5 w-3.5 rounded border-[#8c6d46] text-[#8c6d46] focus:ring-0 cursor-pointer"
            />
          </div>

          {/* Truth Overlay (Simulation only) Toggle */}
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="truth-toggle" className="text-[#0284c7] font-semibold cursor-pointer select-none">
              {t.map.truth_overlay}
            </label>
            <input
              id="truth-toggle"
              type="checkbox"
              checked={showTruthOverlay}
              onChange={onToggleTruthOverlay}
              className="h-3.5 w-3.5 rounded border-[#0284c7] text-[#0284c7] focus:ring-0 cursor-pointer"
            />
          </div>

          {/* Debug Frost Pocket Toggle */}
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="frost-debug-toggle" className="text-[#6b5233] cursor-pointer select-none">
              {t.map.debug_frost_pockets}
            </label>
            <input
              id="frost-debug-toggle"
              type="checkbox"
              checked={showFrostPocketsDebug}
              onChange={onToggleFrostPocketsDebug}
              className="h-3.5 w-3.5 rounded border-[#8c6d46] text-[#0284c7] focus:ring-0 cursor-pointer"
            />
          </div>
        </div>

        {/* Truth overlay explanatory badge */}
        {showTruthOverlay && (
          <div className="p-1.5 rounded bg-[#0284c7]/10 border border-[#0284c7]/30 text-[9px] font-sans text-[#0284c7] leading-tight">
            {t.map.truth_disclaimer}
          </div>
        )}
      </div>
    </div>
  );
}
