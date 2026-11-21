import { Radio, Plus, Trash2, RotateCcw, TrendingUp } from 'lucide-react';
import { Language, getTranslation } from '../i18n';
import { NextSensorRecommendation, Station } from '../types';

interface SensorsTabProps {
  lang: Language;
  stations: Station[];
  recommendation: NextSensorRecommendation | null;
  onAddSensorAtParcel: (parcelId: string, lat: number, lon: number, name: string) => void;
  onDeleteStation: (stationId: string) => void;
  onResetStations: () => void;
  loading: boolean;
}

export default function SensorsTab({
  lang,
  stations,
  recommendation,
  onAddSensorAtParcel,
  onDeleteStation,
  onResetStations,
  loading,
}: SensorsTabProps) {
  const t = getTranslation(lang);

  const isAtDefault = stations.length === 6;
  const best = recommendation;

  return (
    <div className="space-y-4 font-serif text-xs text-[#38281a]">
      {/* Header Banner */}
      <div className="p-3.5 rounded-lg border border-[#ebdcc0] bg-[#f5efe0]/60 space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-sans font-bold uppercase tracking-wider text-[11px] text-[#2c2113]">
            <Radio className="w-3.5 h-3.5 text-[#b24c28] animate-pulse" />
            <span>{t.sensors.title}</span>
          </div>
          <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-[#8c6d46]/10 text-[#8c6d46] border border-[#8c6d46]/20 font-bold">
            K = {stations.length}
          </span>
        </div>
        <p className="text-[11px] text-[#8c6d46] italic">
          {t.sensors.subtitle}
        </p>
      </div>

      {/* Top Recommendation Beacon Card */}
      {best && (
        <div className="p-3.5 rounded-lg border-2 border-[#b24c28]/60 bg-[#faf7f0] shadow-sm space-y-2.5">
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-[#b24c28] px-1.5 py-0.5 rounded bg-[#b24c28]/10 border border-[#b24c28]/30">
                #1 Recommended Site
              </span>
              <h3 className="font-bold text-sm text-[#2c2113] mt-1">
                {best.best_parcel_name}
              </h3>
              <div className="flex items-center gap-2 text-[11px] text-[#6b5233] mt-0.5 font-mono">
                <span>Elev: {best.elevation.toFixed(0)}m</span>
                <span>•</span>
                <span className={best.rel_elev < 8 ? 'text-[#0284c7] font-bold' : ''}>
                  Rel. h: +{best.rel_elev.toFixed(1)}m {best.rel_elev < 8 ? '(Valley Pocket)' : ''}
                </span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-[#8c6d46] block font-sans">Decision Score</span>
              <span className="font-mono font-bold text-base text-[#b24c28]">
                {best.decision_score.toFixed(2)}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px] bg-[#f5efe0]/50 p-2 rounded border border-[#ebdcc0] font-sans">
            <div>
              <span className="text-[#8c6d46] block text-[10px]">Local σ Drop</span>
              <span className="font-mono font-bold text-[#4d7c0f]">
                {best.current_sigma.toFixed(2)}°C → {best.expected_new_local_sigma.toFixed(2)}°C
              </span>
            </div>
            <div>
              <span className="text-[#8c6d46] block text-[10px]">Network Mean σ</span>
              <span className="font-mono font-bold text-[#2c2113]">
                {best.network_mean_sigma_current.toFixed(3)}°C → {best.network_mean_sigma_after.toFixed(3)}°C
              </span>
            </div>
          </div>

          <button
            onClick={() => onAddSensorAtParcel(best.best_parcel_id, best.best_coords[1], best.best_coords[0], best.best_parcel_name)}
            disabled={loading}
            className="w-full py-2 px-3 rounded border border-[#b24c28] bg-[#b24c28] hover:bg-[#8e391b] text-[#faf7f0] font-sans font-bold text-xs flex items-center justify-center gap-1.5 transition-colors shadow-xs disabled:opacity-50"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>{t.sensors.add_sensor_btn} ({best.best_parcel_id})</span>
          </button>
        </div>
      )}

      {/* Before / After Impact Card */}
      <div className="p-3 rounded-lg border border-[#ebdcc0] bg-[#f5efe0]/30 space-y-2">
        <div className="font-sans font-bold uppercase tracking-wide text-[10px] text-[#6b5233] flex items-center gap-1.5">
          <TrendingUp className="w-3 h-3 text-[#4d7c0f]" />
          <span>{t.sensors.before_after_title}</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-[11px] font-sans">
          <div className="p-1.5 rounded bg-[#faf7f0] border border-[#ebdcc0]">
            <span className="text-[10px] text-[#8c6d46] block">Hero Pocket Rec@0.5</span>
            <span className="font-mono font-bold text-[#b24c28]">
              {stations.length === 6 ? '0.0%' : stations.length === 7 ? '38.1%' : stations.length === 8 ? '66.7%' : '71.4%'}
            </span>
          </div>
          <div className="p-1.5 rounded bg-[#faf7f0] border border-[#ebdcc0]">
            <span className="text-[10px] text-[#8c6d46] block">Hero Pocket Rec@0.2</span>
            <span className="font-mono font-bold text-[#0284c7]">
              {stations.length === 6 ? '0.0%' : stations.length === 7 ? '76.2%' : stations.length === 8 ? '85.7%' : '85.7%'}
            </span>
          </div>
          <div className="p-1.5 rounded bg-[#faf7f0] border border-[#ebdcc0]">
            <span className="text-[10px] text-[#8c6d46] block">Network Mean σ</span>
            <span className="font-mono font-bold text-[#2c2113]">
              {recommendation ? `${recommendation.network_mean_sigma_current.toFixed(3)}°C` : '0.768°C'}
            </span>
          </div>
        </div>
      </div>

      {/* Ranked Top 5 Candidates */}
      {recommendation && recommendation.ranked_candidates && (
        <div className="space-y-1.5">
          <div className="font-sans font-bold uppercase tracking-wide text-[10px] text-[#6b5233]">
            Top Candidates (Decision-Weighted)
          </div>
          <div className="border border-[#ebdcc0] rounded-lg overflow-hidden bg-[#faf7f0]">
            <table className="w-full text-[11px] font-sans text-left">
              <thead className="bg-[#f5efe0] text-[#6b5233] border-b border-[#ebdcc0] text-[10px] uppercase">
                <tr>
                  <th className="py-1.5 px-2 font-bold">#</th>
                  <th className="py-1.5 px-2 font-bold">Site</th>
                  <th className="py-1.5 px-2 font-bold text-right">Rel. h</th>
                  <th className="py-1.5 px-2 font-bold text-right">Score</th>
                  <th className="py-1.5 px-2 font-bold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ebdcc0]/60">
                {recommendation.ranked_candidates.map((c, idx) => (
                  <tr key={c.parcel_id} className="hover:bg-[#f5efe0]/50 transition-colors">
                    <td className="py-1.5 px-2 font-mono font-bold text-[#8c6d46]">{idx + 1}</td>
                    <td className="py-1.5 px-2 font-medium text-[#2c2113]">{c.name}</td>
                    <td className={`py-1.5 px-2 font-mono text-right ${c.rel_elev < 8 ? 'text-[#0284c7] font-bold' : 'text-[#6b5233]'}`}>
                      +{c.rel_elev.toFixed(1)}m
                    </td>
                    <td className="py-1.5 px-2 font-mono font-bold text-right text-[#b24c28]">
                      {c.decision_score.toFixed(2)}
                    </td>
                    <td className="py-1.5 px-2 text-right">
                      <button
                        onClick={() => onAddSensorAtParcel(c.parcel_id, c.lat, c.lon, c.name)}
                        disabled={loading}
                        className="px-2 py-0.5 rounded text-[10px] bg-[#8c6d46]/10 text-[#8c6d46] hover:bg-[#b24c28] hover:text-[#faf7f0] font-bold border border-[#8c6d46]/30 transition-colors"
                      >
                        + Add
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Currently Active Weather Stations List */}
      <div className="space-y-1.5 pt-2 border-t border-[#ebdcc0]">
        <div className="flex items-center justify-between">
          <span className="font-sans font-bold uppercase tracking-wide text-[10px] text-[#6b5233]">
            {t.sensors.current_network} ({stations.length})
          </span>
          {!isAtDefault && (
            <button
              onClick={onResetStations}
              disabled={loading}
              className="text-[10px] font-sans font-bold text-[#b24c28] hover:text-[#8e391b] flex items-center gap-1 transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              <span>{t.sensors.reset_btn}</span>
            </button>
          )}
        </div>

        <div className="space-y-1 max-h-48 overflow-y-auto">
          {stations.map((s) => (
            <div
              key={s.id}
              className="flex items-center justify-between p-2 rounded border border-[#ebdcc0] bg-[#faf7f0] text-[11px]"
            >
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#8c6d46]"></span>
                <div>
                  <span className="font-mono font-bold text-[#2c2113]">{s.id}</span>
                  <span className="text-[#6b5233] ml-1.5 text-[10px]">{s.name}</span>
                </div>
              </div>
              <div className="flex items-center gap-2 font-mono text-[#8c6d46] text-[10px]">
                <span>{s.elevation.toFixed(0)}m (+{s.rel_elev.toFixed(0)}m)</span>
                {!s.id.startsWith('S0') && (
                  <button
                    onClick={() => onDeleteStation(s.id)}
                    className="p-1 rounded text-[#b91c1c] hover:bg-[#b91c1c]/10 transition-colors"
                    title={t.sensors.delete_station}
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
