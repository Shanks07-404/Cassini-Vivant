import { useState, useEffect, useRef } from 'react';
import CassiniMap from './components/CassiniMap';
import ParcelCard from './components/ParcelCard';
import ReplayControls from './components/ReplayControls';
import SensorsTab from './components/SensorsTab';
import RoiCalculator from './components/RoiCalculator';
import EvidenceTab from './components/EvidenceTab';
import AssistantTab from './components/AssistantTab';
import {
  ParcelProperties,
  Station,
  ParcelPrediction,
  PredictionResponse,
  NextSensorRecommendation,
  ReplayData,
} from './types';
import { getTranslation, Language } from './i18n';

export default function App() {
  const [lang, setLang] = useState<Language>('en');
  const [selectedParcel, setSelectedParcel] = useState<ParcelProperties | null>(null);
  const [activeTab, setActiveTab] = useState<'risk' | 'sensors' | 'roi' | 'evidence' | 'assistant'>('risk');

  // Replay State
  const [selectedNight, setSelectedNight] = useState<string>('night_001');
  const [currentHour, setCurrentHour] = useState<number>(4);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  // Model & Network State
  const [stations, setStations] = useState<Station[]>([]);
  const [predictions, setPredictions] = useState<Record<string, ParcelPrediction>>({});
  const [recommendation, setRecommendation] = useState<NextSensorRecommendation | null>(null);
  const [meshGeoJSON, setMeshGeoJSON] = useState<any | null>(null);
  const [truthFrozenParcelIds, setTruthFrozenParcelIds] = useState<Set<string>>(new Set());
  const [regionalTemp, setRegionalTemp] = useState<number>(2.4);
  const [allStationsSafe, setAllStationsSafe] = useState<boolean>(true);
  const [loadingAction, setLoadingAction] = useState<boolean>(false);

  // Map Toggles
  const [showMesh, setShowMesh] = useState<boolean>(true);
  const [showUncertainty, setShowUncertainty] = useState<boolean>(false);
  const [showTruthOverlay, setShowTruthOverlay] = useState<boolean>(false);
  const [showFrostPocketsDebug, setShowFrostPocketsDebug] = useState<boolean>(false);

  const t = getTranslation(lang);

  // 1. Fetch Active Stations & Mesh
  const refreshStationsAndMesh = () => {
    fetch('/api/sensors')
      .then((res) => res.json())
      .then((data) => {
        setStations(data.stations || []);
      })
      .catch((err) => console.error('Failed to load stations:', err));

    fetch('/api/mesh')
      .then((res) => res.json())
      .then((data) => {
        setMeshGeoJSON(data);
      })
      .catch((err) => console.error('Failed to load mesh:', err));
  };

  useEffect(() => {
    refreshStationsAndMesh();
  }, []);

  // 2. Fetch Replay Data (Ground truth and regional temp) for current night and hour
  useEffect(() => {
    fetch(`/api/replay/${selectedNight}`)
      .then((res) => res.json())
      .then((data: ReplayData) => {
        const frame = data.frames.find((f) => f.hour === currentHour) || data.frames[8];
        if (frame) {
          setRegionalTemp(frame.regional_temp);
          // Ground truth frozen parcels
          const frozenSet = new Set<string>();
          Object.entries(frame.parcel_temperatures).forEach(([pid, temp]) => {
            if (temp < 0.0) frozenSet.add(pid);
          });
          setTruthFrozenParcelIds(frozenSet);

          // Check if all active sensors read > 0°C
          const readings = Object.values(frame.sensor_readings);
          const allAboveZero = readings.length > 0 && readings.every((r) => r > 0.0);
          setAllStationsSafe(allAboveZero);
        }
      })
      .catch((err) => console.error('Failed to load replay frame:', err));
  }, [selectedNight, currentHour]);

  // 3. Fetch Live GP Predictions & Next Sensor Recommendation
  const refreshPredictionsAndRecommendation = () => {
    fetch(`/api/predict?night=${selectedNight}&hour=${currentHour}`)
      .then((res) => res.json())
      .then((data: PredictionResponse) => {
        const predMap: Record<string, ParcelPrediction> = {};
        data.predictions.forEach((p) => {
          predMap[p.parcel_id] = p;
        });
        setPredictions(predMap);
      })
      .catch((err) => console.error('Failed to load predictions:', err));

    fetch(`/api/next-sensor?night=${selectedNight}&hour=${currentHour}`, {
      method: 'POST',
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'success') {
          setRecommendation(data.recommendation);
        }
      })
      .catch((err) => console.error('Failed to load next sensor recommendation:', err));
  };

  useEffect(() => {
    refreshPredictionsAndRecommendation();
  }, [selectedNight, currentHour, stations.length]);

  // 4. Replay Timer Loop
  const HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8];
  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = window.setInterval(() => {
        setCurrentHour((prev) => {
          const idx = HOURS.indexOf(prev);
          const nextIdx = (idx + 1) % HOURS.length;
          return HOURS[nextIdx];
        });
      }, 1400);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying]);

  // 5. Sensor Management Actions
  const handleAddSensorAtParcel = (parcelId: string, lat: number, lon: number, name: string) => {
    setLoadingAction(true);
    fetch('/api/sensors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat,
        lon,
        name: `Station ${parcelId} (${name.split('(')[1]?.replace(')', '') || 'Coteau'})`,
      }),
    })
      .then((res) => res.json())
      .then(() => {
        refreshStationsAndMesh();
        setLoadingAction(false);
      })
      .catch((err) => {
        console.error('Failed to add station:', err);
        setLoadingAction(false);
      });
  };

  const handleDeleteStation = (stationId: string) => {
    setLoadingAction(true);
    fetch(`/api/sensors/${stationId}`, { method: 'DELETE' })
      .then((res) => res.json())
      .then(() => {
        refreshStationsAndMesh();
        setLoadingAction(false);
      })
      .catch((err) => {
        console.error('Failed to delete station:', err);
        setLoadingAction(false);
      });
  };

  const handleResetStations = () => {
    setLoadingAction(true);
    fetch('/api/sensors/reset', { method: 'POST' })
      .then((res) => res.json())
      .then(() => {
        refreshStationsAndMesh();
        setLoadingAction(false);
      })
      .catch((err) => {
        console.error('Failed to reset stations:', err);
        setLoadingAction(false);
      });
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#faf7f0] text-[#38281a] select-none font-serif">
      {/* Top Bar */}
      <header className="h-14 border-b border-[#ebdcc0] bg-[#f5efe0] px-5 flex items-center justify-between shadow-sm z-20">
        {/* Left: Logo & Tagline */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded border border-[#8c6d46] bg-[#8c6d46] text-[#faf7f0] font-serif font-bold flex items-center justify-center text-lg shadow-sm">
            C
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-serif font-bold text-base tracking-wide text-[#2c2113]">
                {t.app.title}
              </h1>
              <span className="text-[10px] uppercase font-sans tracking-widest px-1.5 py-0.5 rounded bg-[#8c6d46]/10 text-[#8c6d46] font-semibold border border-[#8c6d46]/20">
                1747 • 2026
              </span>
            </div>
            <p className="text-xs text-[#8c6d46] italic -mt-0.5 font-serif">
              {t.app.tagline}
            </p>
          </div>
        </div>

        {/* Right: Badges & Controls */}
        <div className="flex items-center gap-3">
          {/* Permanent SIMULATED DATA badge (Section 13) */}
          <div className="px-2.5 py-1 text-xs font-sans font-bold uppercase tracking-wider rounded border border-[#b24c28]/50 bg-[#b24c28]/10 text-[#b24c28] flex items-center gap-1.5 shadow-xs">
            <span className="w-2 h-2 rounded-full bg-[#b24c28] animate-pulse"></span>
            <span>{t.app.simulated_badge}</span>
          </div>

          {/* Language Toggle */}
          <button
            onClick={() => setLang(lang === 'en' ? 'fr' : 'en')}
            className="px-2.5 py-1 text-xs font-sans font-semibold rounded border border-[#8c6d46]/60 bg-[#faf7f0] text-[#4a3821] hover:bg-[#ebdcc0] transition-colors shadow-xs"
            title="Toggle Language / Changer la langue"
          >
            {lang === 'en' ? 'FR' : 'EN'}
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left 70%: Map + Replay Bar */}
        <div className="w-[70%] h-full flex flex-col relative border-r border-[#ebdcc0]">
          <div className="flex-1 relative">
            <CassiniMap
              lang={lang}
              selectedParcel={selectedParcel}
              onSelectParcel={setSelectedParcel}
              stations={stations}
              predictions={predictions}
              meshGeoJSON={meshGeoJSON}
              showMesh={showMesh}
              onToggleMesh={() => setShowMesh(!showMesh)}
              showUncertainty={showUncertainty}
              onToggleUncertainty={() => setShowUncertainty(!showUncertainty)}
              showTruthOverlay={showTruthOverlay}
              onToggleTruthOverlay={() => setShowTruthOverlay(!showTruthOverlay)}
              truthFrozenParcelIds={truthFrozenParcelIds}
              showFrostPocketsDebug={showFrostPocketsDebug}
              onToggleFrostPocketsDebug={() => setShowFrostPocketsDebug(!showFrostPocketsDebug)}
              recommendation={recommendation}
              onAddSensorAtParcel={handleAddSensorAtParcel}
            />
          </div>

          {/* Replay Scrubber Control Bar */}
          <ReplayControls
            lang={lang}
            selectedNight={selectedNight}
            onSelectNight={setSelectedNight}
            currentHour={currentHour}
            onHourChange={setCurrentHour}
            isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(!isPlaying)}
            regionalTemp={regionalTemp}
            allStationsSafe={allStationsSafe}
          />
        </div>

        {/* Right 30%: Dashboard Panel */}
        <div className="w-[30%] h-full flex flex-col bg-[#faf7f0] border-l border-[#ebdcc0]/60 overflow-hidden">
          {/* Navigation Tabs */}
          <nav className="flex border-b border-[#ebdcc0] bg-[#f5efe0]/80 px-2 pt-2 gap-1 overflow-x-auto text-xs font-sans">
            {(['risk', 'sensors', 'roi', 'evidence', 'assistant'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-2 rounded-t font-medium transition-colors border-t border-x -mb-px ${
                  activeTab === tab
                    ? 'bg-[#faf7f0] border-[#ebdcc0] text-[#2c2113] font-bold shadow-xs'
                    : 'border-transparent text-[#8c6d46] hover:text-[#2c2113] hover:bg-[#ebdcc0]/40'
                }`}
              >
                {t.nav.tabs[tab]}
              </button>
            ))}
          </nav>

          {/* Tab Content Body */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {/* 1. Risk Analysis Tab */}
            {activeTab === 'risk' && (
              <div className="space-y-4">
                <ParcelCard
                  parcel={selectedParcel}
                  onClose={() => setSelectedParcel(null)}
                  lang={lang}
                />

                <div className="p-4 rounded-lg border border-[#ebdcc0] bg-[#f5efe0]/30 font-serif text-xs text-[#8c6d46] space-y-2">
                  <div className="font-bold text-[#4a3821] uppercase tracking-wide text-[11px] font-sans">
                    Cadastre Synthétique (4 km x 4 km)
                  </div>
                  <p>
                    {lang === 'en'
                      ? 'Live Universal Kriging GP microclimate estimation. Predictions update in real time with active weather stations.'
                      : 'Estimation microclimatique par Krigeage Universel GP en temps réel, recalibrée dynamiquement selon les stations météo actives.'}
                  </p>
                </div>
              </div>
            )}

            {/* 2. Optimal Sensors Tab (Where to measure next) */}
            {activeTab === 'sensors' && (
              <SensorsTab
                lang={lang}
                stations={stations}
                recommendation={recommendation}
                onAddSensorAtParcel={handleAddSensorAtParcel}
                onDeleteStation={handleDeleteStation}
                onResetStations={handleResetStations}
                loading={loadingAction}
              />
            )}

            {/* 3. ROI Calculator Tab */}
            {activeTab === 'roi' && (
              <RoiCalculator
                lang={lang}
                selectedNight={selectedNight}
              />
            )}

            {/* 4. Model Evidence Tab */}
            {activeTab === 'evidence' && (
              <EvidenceTab
                lang={lang}
              />
            )}

            {/* 5. Assistant Tab */}
            {activeTab === 'assistant' && (
              <AssistantTab
                lang={lang}
                selectedParcel={selectedParcel}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
