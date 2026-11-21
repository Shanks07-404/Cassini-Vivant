import { ParcelProperties } from '../types';
import { getTranslation, Language } from '../i18n';

interface ParcelCardProps {
  parcel: ParcelProperties | null;
  onClose: () => void;
  lang: Language;
}

export default function ParcelCard({ parcel, onClose, lang }: ParcelCardProps) {
  const t = getTranslation(lang);

  if (!parcel) {
    return (
      <div className="p-6 border border-dashed border-[#ebdcc0] rounded-lg bg-[#f5efe0]/40 text-center font-serif text-[#8c6d46]">
        <div className="w-10 h-10 mx-auto mb-2 rounded-full border border-[#ebdcc0] flex items-center justify-center text-[#8c6d46]">
          ✦
        </div>
        <p className="text-sm italic">{t.parcel.select_prompt}</p>
      </div>
    );
  }

  const riskPercent = Math.round(parcel.value * 100);
  const isHighRisk = parcel.value > 0.5;

  return (
    <div className="rounded-lg border border-[#8c6d46]/40 bg-[#faf7f0] shadow-md overflow-hidden font-serif">
      {/* Header */}
      <div className="bg-[#f5efe0] px-4 py-3 border-b border-[#ebdcc0] flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-[#8c6d46]/15 text-[#4a3821]">
              {parcel.id}
            </span>
            <h3 className="font-bold text-base text-[#2c2113]">{parcel.name}</h3>
          </div>
          <div className="text-xs text-[#8c6d46] mt-0.5">
            AOP Démo (Terroir Fictif) • {parcel.is_frost_pocket ? t.parcel.frost_pocket_badge : t.parcel.slope_parcel_badge}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-[#8c6d46] hover:text-[#2c2113] p-1 rounded hover:bg-[#ebdcc0] transition-colors text-xs font-sans"
          aria-label={t.parcel.close}
        >
          ✕
        </button>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Risk meter */}
        <div className="p-3 rounded border border-[#ebdcc0] bg-[#f5efe0]/60">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-semibold text-[#4a3821]">{t.parcel.risk_score}</span>
            <span
              className={`font-mono text-sm font-bold ${
                isHighRisk ? 'text-[#b24c28]' : 'text-[#4d7c0f]'
              }`}
            >
              {riskPercent}%
            </span>
          </div>
          <div className="w-full h-2 rounded-full bg-[#ebdcc0] overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                isHighRisk ? 'bg-[#b24c28]' : 'bg-[#4d7c0f]'
              }`}
              style={{ width: `${riskPercent}%` }}
            />
          </div>
          <div className="mt-2 text-[11px] font-sans text-[#8c6d46]">
            {isHighRisk ? t.parcel.status_critical : t.parcel.status_safe}
          </div>
        </div>

        {/* Topographic parameters */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded border border-[#ebdcc0] bg-white/40">
            <div className="text-[11px] text-[#8c6d46]">{t.parcel.elevation}</div>
            <div className="font-mono font-bold text-sm text-[#2c2113]">{parcel.elevation} m</div>
          </div>
          <div className="p-2 rounded border border-[#ebdcc0] bg-white/40">
            <div className="text-[11px] text-[#8c6d46]">{t.parcel.rel_elevation}</div>
            <div className="font-mono font-bold text-sm text-[#2c2113]">+{parcel.rel_elev} m</div>
          </div>
          <div className="p-2 rounded border border-[#ebdcc0] bg-white/40">
            <div className="text-[11px] text-[#8c6d46]">{t.parcel.slope}</div>
            <div className="font-mono font-bold text-sm text-[#2c2113]">{parcel.slope_deg}°</div>
          </div>
          <div className="p-2 rounded border border-[#ebdcc0] bg-white/40">
            <div className="text-[11px] text-[#8c6d46]">{t.parcel.area}</div>
            <div className="font-mono font-bold text-sm text-[#2c2113]">{parcel.area_ha} ha</div>
          </div>
        </div>

        {parcel.is_frost_pocket && (
          <div className="p-2.5 rounded border border-[#0284c7]/30 bg-[#0284c7]/10 text-xs text-[#0369a1] font-sans flex items-center gap-2">
            <span className="text-base">❄</span>
            <span>
              <strong>Attention :</strong> {t.map.frost_pocket_legend}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
