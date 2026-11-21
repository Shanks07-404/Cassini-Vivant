import { Play, Pause, Moon } from 'lucide-react';
import { Language, getTranslation } from '../i18n';

interface ReplayControlsProps {
  lang: Language;
  selectedNight: string;
  onSelectNight: (nightId: string) => void;
  currentHour: number;
  onHourChange: (hour: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  regionalTemp: number;
  allStationsSafe: boolean;
}

const HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8];

export default function ReplayControls({
  lang,
  selectedNight,
  onSelectNight,
  currentHour,
  onHourChange,
  isPlaying,
  onTogglePlay,
  regionalTemp,
  allStationsSafe,
}: ReplayControlsProps) {
  const t = getTranslation(lang);

  const getHourLabel = (h: number) => {
    const s = `${h.toString().padStart(2, '0')}:00`;
    if (h === 4) return `${s} (Peak Inversion)`;
    if (h === 20) return `${s} (Sunset)`;
    if (h === 8) return `${s} (Sunrise)`;
    return s;
  };

  return (
    <div className="flex flex-col bg-[#faf7f0]/95 backdrop-blur-sm border-t border-[#ebdcc0] px-5 py-3 text-xs font-serif shadow-md select-none">
      {/* Top row: Night selector, regional temp, and inversion indicator */}
      <div className="flex items-center justify-between gap-4 mb-2">
        {/* Night selector */}
        <div className="flex items-center gap-2">
          <span className="font-bold text-[#2c2113]">{t.replay.select_night}</span>
          <select
            value={selectedNight}
            onChange={(e) => onSelectNight(e.target.value)}
            className="px-2.5 py-1 text-xs font-serif font-semibold rounded border border-[#8c6d46]/50 bg-[#faf7f0] text-[#2c2113] focus:outline-none focus:ring-1 focus:ring-[#8c6d46]"
          >
            <option value="night_001">night_001 — {t.replay.hero_label}</option>
            {Array.from({ length: 59 }, (_, i) => {
              const nid = `night_${(i + 2).toString().padStart(3, '0')}`;
              return (
                <option key={nid} value={nid}>
                  {nid} (T_min ~ {-(i % 4) + 1}°C)
                </option>
              );
            })}
          </select>
        </div>

        {/* Regional Conditions & Station status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#f5efe0] border border-[#ebdcc0] text-[11px] font-sans">
            <Moon className="w-3.5 h-3.5 text-[#8c6d46]" />
            <span className="text-[#8c6d46] font-medium">{t.replay.regional_t}</span>
            <span className={`font-mono font-bold ${regionalTemp <= 0 ? 'text-[#b91c1c]' : 'text-[#4d7c0f]'}`}>
              {regionalTemp > 0 ? `+${regionalTemp.toFixed(1)}` : regionalTemp.toFixed(1)}°C
            </span>
          </div>

          {selectedNight === 'night_001' && allStationsSafe && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#4d7c0f]/10 border border-[#4d7c0f]/30 text-[#4d7c0f] text-[11px] font-sans font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-[#4d7c0f]"></span>
              <span>{t.map.all_stations_safe}</span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom row: Time scrubber & Play/pause */}
      <div className="flex items-center gap-3">
        {/* Play/Pause Button */}
        <button
          onClick={onTogglePlay}
          className="p-2 rounded-full border border-[#8c6d46]/50 bg-[#8c6d46] text-[#faf7f0] hover:bg-[#6b5233] transition-colors shadow-xs flex items-center justify-center shrink-0"
          title={isPlaying ? t.replay.pause : t.replay.play}
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
        </button>

        {/* Hour stepper pills */}
        <div className="flex-1 flex items-center gap-1 overflow-x-auto py-1">
          {HOURS.map((h) => {
            const isActive = currentHour === h;
            return (
              <button
                key={h}
                onClick={() => onHourChange(h)}
                className={`flex-1 min-w-[42px] py-1 px-1 rounded text-center font-mono text-[11px] transition-all border ${
                  isActive
                    ? 'bg-[#2c2113] text-[#faf7f0] border-[#2c2113] font-bold shadow-xs scale-105'
                    : 'bg-[#f5efe0]/80 text-[#6b5233] border-[#ebdcc0] hover:bg-[#ebdcc0]/50'
                } ${h === 4 ? 'border-dashed border-[#b91c1c]/70' : ''}`}
                title={getHourLabel(h)}
              >
                {h.toString().padStart(2, '0')}h
              </button>
            );
          })}
        </div>

        {/* Current label */}
        <div className="hidden md:block min-w-[130px] text-right font-mono text-xs text-[#2c2113] font-bold">
          {getHourLabel(currentHour)}
        </div>
      </div>
    </div>
  );
}
