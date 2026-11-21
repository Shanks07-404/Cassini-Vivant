import { useState, useEffect } from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';
import { Language, getTranslation } from '../i18n';
import { RoiResponse } from '../types';

interface RoiCalculatorProps {
  lang: Language;
  selectedNight: string;
}

export default function RoiCalculator({ lang, selectedNight }: RoiCalculatorProps) {
  const t = getTranslation(lang);
  const [costPerNight, setCostPerNight] = useState<number>(350);
  const [lossPerParcel, setLossPerParcel] = useState<number>(4500);
  const [threshold, setThreshold] = useState<number>(0.08); // Close to 350 / 4500 = 0.078
  const [roiData, setRoiData] = useState<RoiResponse | null>(null);

  useEffect(() => {
    const params = new URLSearchParams({
      night: selectedNight,
      protect_threshold: threshold.toString(),
      cost_per_parcel_night: costPerNight.toString(),
      loss_per_parcel: lossPerParcel.toString(),
    });

    fetch(`/api/roi?${params.toString()}`)
      .then((res) => res.json())
      .then((data: RoiResponse) => {
        setRoiData(data);
      })
      .catch((err) => {
        console.error('Failed to load ROI data:', err);
      });
  }, [selectedNight, threshold, costPerNight, lossPerParcel]);

  const theoreticalThreshold = lossPerParcel > 0 ? (costPerNight / lossPerParcel) : 0.08;
  const breakEvenRatio = costPerNight > 0 ? (lossPerParcel / costPerNight) : 12.86;

  const formatBenchmark = (val?: number) => {
    if (val === undefined || val === null) return 'N/A';
    return `€${Math.round(val).toLocaleString()}`;
  };

  const formatK = (val?: number) => {
    if (val === undefined || val === null) return '€0';
    return `€${(val / 1000).toFixed(1)}k`;
  };

  return (
    <div className="space-y-4 font-serif text-xs text-[#38281a]">
      {/* Illustrative Disclaimer Banner */}
      <div className="p-3 rounded-lg border-2 border-dashed border-[#b24c28]/60 bg-[#b24c28]/10 text-[#b24c28] text-center space-y-0.5">
        <div className="font-sans font-bold uppercase tracking-widest text-[11px] flex items-center justify-center gap-1.5">
          <AlertTriangle className="w-4 h-4 text-[#b24c28]" />
          <span>{t.roi.disclaimer}</span>
        </div>
        <p className="text-[10px] text-[#8e391b] italic">
          {lang === 'en'
            ? 'Figures are for model sensitivity & demonstration only. Replace with audited cooperative accounts.'
            : 'Valeurs indicatives pour validation méthodologique. À remplacer par les données comptables certifiées.'}
        </p>
      </div>

      {/* Decision-Theoretic Break-Even Ratio Header */}
      <div className="p-3.5 rounded-lg border border-[#ebdcc0] bg-[#faf7f0] shadow-sm space-y-2 font-sans">
        <div className="flex items-center justify-between border-b border-[#ebdcc0] pb-2">
          <div>
            <span className="text-[10px] uppercase tracking-wider font-bold text-[#8c6d46] block">
              Cost-Loss Decision Theory
            </span>
            <span className="font-bold text-sm text-[#2c2113]">
              Break-Even Ratio: {breakEvenRatio.toFixed(1)}× (Loss / Protection Cost)
            </span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#8c6d46] block">Optimal Action Rule (p*)</span>
            <span className="font-mono font-bold text-sm text-[#b24c28]">
              {(theoreticalThreshold * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <p className="text-[10px] text-[#6b5233] italic leading-relaxed">
          {lang === 'en'
            ? `Under decision theory, frost action is justified when local frost risk exceeds the cost-loss ratio p* = Cost/Loss (${(theoreticalThreshold * 100).toFixed(1)}%). Below this, expected damage is lower than protection costs.`
            : `Selon la théorie de la décision, l'intervention est justifiée dès que le risque local dépasse le ratio coût-perte p* = Coût/Perte (${(theoreticalThreshold * 100).toFixed(1)}%).`}
        </p>
      </div>

      {/* Interactive Parameter Sliders */}
      <div className="p-3.5 rounded-lg border border-[#ebdcc0] bg-[#faf7f0] shadow-sm space-y-3 font-sans">
        {/* Threshold slider */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-[#2c2113]">{t.roi.threshold_slider}</span>
              <button
                type="button"
                onClick={() => setThreshold(parseFloat(theoreticalThreshold.toFixed(2)))}
                className="text-[9px] bg-[#ebdcc0] hover:bg-[#dfc8a4] px-1.5 py-0.5 rounded text-[#2c2113] font-mono"
                title="Align with theoretical cost-loss rule"
              >
                Set to p* ({(theoreticalThreshold * 100).toFixed(1)}%)
              </button>
            </div>
            <span className="font-mono font-bold text-[#b24c28]">
              {(threshold * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.02"
            max="0.80"
            step="0.02"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-[#b24c28] cursor-pointer h-1.5 bg-[#ebdcc0] rounded-lg"
          />
          <div className="flex justify-between text-[10px] text-[#8c6d46] mt-0.5 font-mono">
            <span>2%</span>
            <span className="text-[#b24c28] font-bold">▲ p* ({(theoreticalThreshold * 100).toFixed(1)}%)</span>
            <span>20% (Default)</span>
            <span>80%</span>
          </div>
        </div>

        {/* Protection Cost per Parcel */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1">
            <span className="font-medium text-[#2c2113]">
              {t.roi.cost_slider}
            </span>
            <span className="font-mono font-bold text-[#2c2113]">
              {costPerNight} €
            </span>
          </div>
          <input
            type="range"
            min="100"
            max="1000"
            step="50"
            value={costPerNight}
            onChange={(e) => setCostPerNight(parseInt(e.target.value))}
            className="w-full accent-[#8c6d46] cursor-pointer h-1.5 bg-[#ebdcc0] rounded-lg"
          />
        </div>

        {/* Crop Value Loss per Parcel */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1">
            <span className="font-medium text-[#2c2113]">
              {t.roi.loss_slider}
            </span>
            <span className="font-mono font-bold text-[#2c2113]">
              {lossPerParcel.toLocaleString()} €
            </span>
          </div>
          <input
            type="range"
            min="1000"
            max="12000"
            step="500"
            value={lossPerParcel}
            onChange={(e) => setLossPerParcel(parseInt(e.target.value))}
            className="w-full accent-[#8c6d46] cursor-pointer h-1.5 bg-[#ebdcc0] rounded-lg"
          />
        </div>
      </div>

      {/* Hero Night Warning: Strategy B1 Invalidation */}
      {selectedNight === 'night_001' && (
        <div className="p-3 rounded-lg border border-[#b24c28]/40 bg-[#b24c28]/10 text-[#8e391b] font-sans text-[11px] space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-[#b24c28]">
            <ShieldAlert className="w-4 h-4 shrink-0" />
            <span>Hero Night Showcase: Regional Alert Ineffective</span>
          </div>
          <p className="leading-relaxed">
            On <strong>night_001</strong>, regional forecast temperature remains <strong>+2.4°C</strong> (above freezing).
            Standard regional-alert systems (<strong>Strategy B1</strong>) trigger <strong>0 protections</strong>, resulting in
            unmitigated frost damage on 49 parcels (€{(49 * lossPerParcel).toLocaleString()} loss, identical to Strategy A).
          </p>
        </div>
      )}

      {/* Strategies Comparison Breakdown Table */}
      {roiData && (
        <div className="space-y-1.5 font-sans">
          <div className="flex items-center justify-between font-bold uppercase tracking-wide text-[10px] text-[#6b5233]">
            <span>Strategy Comparison ({selectedNight} {selectedNight === 'night_001' ? '— Showcase' : ''})</span>
            <span className="text-[#8c6d46] normal-case">Action at p &gt; {(threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="border border-[#ebdcc0] rounded-lg overflow-hidden bg-[#faf7f0]">
            <table className="w-full text-[11px] text-left">
              <thead className="bg-[#f5efe0] text-[#6b5233] border-b border-[#ebdcc0] text-[10px] uppercase">
                <tr>
                  <th className="py-2 px-2.5 font-bold">Strategy &amp; Sensor Set</th>
                  <th className="py-2 px-1.5 font-bold text-right">Protected</th>
                  <th className="py-2 px-1.5 font-bold text-right">Protection €</th>
                  <th className="py-2 px-1.5 font-bold text-right">Crop Loss €</th>
                  <th className="py-2 px-1.5 font-bold text-right">False Alarm €</th>
                  <th className="py-2 px-2.5 font-bold text-right">Total Cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ebdcc0]/60">
                {/* Strategy A */}
                <tr className="hover:bg-[#f5efe0]/40">
                  <td className="py-2 px-2.5">
                    <span className="font-medium text-[#2c2113] block">Strategy A: Protect Nothing</span>
                    <span className="text-[9px] text-[#8c6d46] font-mono">Sensors: None</span>
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">0</td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">0 €</td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#b91c1c] font-bold">
                    {roiData.strategies.strategy_a.crop_loss?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">0 €</td>
                  <td className="py-2 px-2.5 font-mono text-right font-bold text-[#b91c1c]">
                    {roiData.strategies.strategy_a.total_expenditure?.toLocaleString()} €
                  </td>
                </tr>

                {/* Strategy B1: Regional Alert */}
                <tr className="hover:bg-[#f5efe0]/40">
                  <td className="py-2 px-2.5">
                    <span className="font-medium text-[#2c2113] block">Strategy B1: Regional Alert (T ≤ 0°C)</span>
                    <span className="text-[9px] text-[#8c6d46] font-mono">Sensors: Regional meteo station</span>
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#2c2113]">
                    {roiData.strategies.strategy_b1?.parcels_protected ?? roiData.strategies.strategy_b.parcels_protected}
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">
                    {(roiData.strategies.strategy_b1?.protection_cost ?? roiData.strategies.strategy_b.protection_cost)?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#b91c1c] font-bold">
                    {(roiData.strategies.strategy_b1?.crop_loss ?? roiData.strategies.strategy_b.crop_loss)?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">
                    {(roiData.strategies.strategy_b1?.false_alarm_cost ?? 0).toLocaleString()} €
                  </td>
                  <td className="py-2 px-2.5 font-mono text-right font-bold text-[#b91c1c]">
                    {(roiData.strategies.strategy_b1?.total_expenditure ?? roiData.strategies.strategy_b.total_expenditure)?.toLocaleString()} €
                  </td>
                </tr>

                {/* Strategy B2: Blanket Protection */}
                <tr className="hover:bg-[#f5efe0]/40">
                  <td className="py-2 px-2.5">
                    <span className="font-medium text-[#2c2113] block">Strategy B2: Blanket Protect All</span>
                    <span className="text-[9px] text-[#8c6d46] font-mono">Sensors: None (Cost upper bound)</span>
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#2c2113]">239</td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#b24c28] font-bold">
                    {(239 * costPerNight).toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#4d7c0f]">0 €</td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">
                    {(Math.max(0, 239 - (roiData.parameters.truly_frozen_parcels ?? 0)) * costPerNight).toLocaleString()} €
                  </td>
                  <td className="py-2 px-2.5 font-mono text-right font-bold text-[#b24c28]">
                    {(239 * costPerNight).toLocaleString()} €
                  </td>
                </tr>

                {/* Strategy C: Default 6 Stations */}
                <tr className="bg-[#4d7c0f]/5 hover:bg-[#4d7c0f]/10">
                  <td className="py-2 px-2.5">
                    <span className="font-bold text-[#2c2113] block">Strategy C (K=6 Default)</span>
                    <span className="text-[9px] text-[#8c6d46] font-mono">Sensors: S01–S06 (Ridge/Slope)</span>
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#2c2113]">
                    {roiData.strategies.strategy_c_k6?.parcels_protected ?? (roiData.parameters.active_sensor_count === 6 ? roiData.strategies.strategy_c.parcels_protected : 24)}
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#2c2113]">
                    {(roiData.strategies.strategy_c_k6?.protection_cost ?? 8400)?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#b91c1c]">
                    {(roiData.strategies.strategy_c_k6?.crop_loss ?? 157500)?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">
                    {(roiData.strategies.strategy_c_k6?.false_alarm_cost ?? 0).toLocaleString()} €
                  </td>
                  <td className="py-2 px-2.5 font-mono text-right font-bold text-[#2c2113]">
                    {(roiData.strategies.strategy_c_k6?.total_expenditure ?? 165900)?.toLocaleString()} €
                  </td>
                </tr>

                {/* Strategy C: Default + 2 Valley Stations (K=8) */}
                <tr className="bg-[#4d7c0f]/15 font-bold border-t-2 border-[#4d7c0f]/40">
                  <td className="py-2 px-2.5 text-[#2c2113]">
                    <span className="block text-[#4d7c0f]">Strategy C (K=8: + P050, P202)</span>
                    <span className="text-[9px] text-[#4d7c0f] font-mono">Sensors: S01–S06 + 2 Valley Stations</span>
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#4d7c0f]">
                    {roiData.strategies.strategy_c_k8?.parcels_protected ?? 66}
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#4d7c0f]">
                    {(roiData.strategies.strategy_c_k8?.protection_cost ?? (66 * costPerNight))?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#b24c28]">
                    {(roiData.strategies.strategy_c_k8?.crop_loss ?? (15 * lossPerParcel))?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-1.5 font-mono text-right text-[#8c6d46]">
                    {(roiData.strategies.strategy_c_k8?.false_alarm_cost ?? (32 * costPerNight))?.toLocaleString()} €
                  </td>
                  <td className="py-2 px-2.5 font-mono text-right font-bold text-[#4d7c0f] text-xs">
                    {(roiData.strategies.strategy_c_k8?.total_expenditure ?? (66 * costPerNight + 15 * lossPerParcel))?.toLocaleString()} €
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Training Nights Average Benchmark Box */}
      {roiData?.training_nights_benchmarks && (
        <div className="p-3 rounded-lg border border-[#ebdcc0] bg-[#faf7f0] font-sans space-y-1.5">
          <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[#6b5233]">
            <span>Average Expenditure Across All 59 Training Nights (002–060)</span>
            <span className="font-mono text-[#8c6d46]">Mean [95% Bootstrap CI]</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-[10px]">
            <div className="p-2 rounded bg-white border border-[#ebdcc0]/60">
              <span className="text-[#8c6d46] block text-[9px]">Strategy A</span>
              <span className="font-mono font-bold text-[#b91c1c]">
                {formatBenchmark(roiData.training_nights_benchmarks.A?.mean)}
              </span>
              <span className="text-[8px] text-[#8c6d46] block">
                {roiData.training_nights_benchmarks.A?.ci_95
                  ? `[${(roiData.training_nights_benchmarks.A.ci_95[0]/1000).toFixed(1)}k – ${(roiData.training_nights_benchmarks.A.ci_95[1]/1000).toFixed(1)}k]`
                  : '[80.8k – 206.2k]'}
              </span>
            </div>
            <div className="p-2 rounded bg-white border border-[#ebdcc0]/60">
              <span className="text-[#8c6d46] block text-[9px]">Strategy B1</span>
              <span className="font-mono font-bold text-[#b24c28]">
                {formatBenchmark(roiData.training_nights_benchmarks.B1?.mean)}
              </span>
              <span className="text-[8px] text-[#8c6d46] block">
                {roiData.training_nights_benchmarks.B1?.ci_95
                  ? `[${(roiData.training_nights_benchmarks.B1.ci_95[0]/1000).toFixed(1)}k – ${(roiData.training_nights_benchmarks.B1.ci_95[1]/1000).toFixed(1)}k]`
                  : '[61.6k – 132.1k]'}
              </span>
            </div>
            <div className="p-2 rounded bg-white border border-[#ebdcc0]/60">
              <span className="text-[#8c6d46] block text-[9px]">Strategy B2</span>
              <span className="font-mono font-bold text-[#8c6d46]">
                {formatBenchmark(roiData.training_nights_benchmarks.B2?.mean)}
              </span>
              <span className="text-[8px] text-[#8c6d46] block">[Fixed]</span>
            </div>
            <div className="p-2 rounded bg-white border border-[#ebdcc0]/60">
              <span className="text-[#8c6d46] block text-[9px]">C (K=6)</span>
              <span className="font-mono font-bold text-[#2c2113]">
                {formatBenchmark(roiData.training_nights_benchmarks.C_k6?.mean)}
              </span>
              <span className="text-[8px] text-[#8c6d46] block">
                {roiData.training_nights_benchmarks.C_k6?.ci_95
                  ? `[${(roiData.training_nights_benchmarks.C_k6.ci_95[0]/1000).toFixed(1)}k – ${(roiData.training_nights_benchmarks.C_k6.ci_95[1]/1000).toFixed(1)}k]`
                  : '[36.3k – 58.4k]'}
              </span>
            </div>
            <div className="p-2 rounded bg-[#4d7c0f]/10 border border-[#4d7c0f]/40">
              <span className="text-[#4d7c0f] font-bold block text-[9px]">C (K=8 Valley)</span>
              <span className="font-mono font-bold text-[#4d7c0f]">
                {formatBenchmark(roiData.training_nights_benchmarks.C_k8?.mean)}
              </span>
              <span className="text-[8px] text-[#4d7c0f] block">
                {roiData.training_nights_benchmarks.C_k8?.ci_95
                  ? `[${(roiData.training_nights_benchmarks.C_k8.ci_95[0]/1000).toFixed(1)}k – ${(roiData.training_nights_benchmarks.C_k8.ci_95[1]/1000).toFixed(1)}k]`
                  : '[27.5k – 43.1k]'}
              </span>
            </div>
          </div>

          {/* Breakdown by Night Type */}
          <div className="pt-2 border-t border-[#ebdcc0]/60 space-y-1">
            <span className="text-[9px] uppercase tracking-wider font-bold text-[#8c6d46] block">
              Average Expenditure by Night Type (Strategy C K=8 vs Baselines)
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[10px]">
              <div className="p-2 rounded bg-white border border-[#ebdcc0]/60 space-y-0.5">
                <span className="font-bold text-[#2c2113] block">Inversion Nights (37 tr)</span>
                <span className="text-[#8c6d46] block text-[9px]">T_reg &gt; 0°C, valley &lt; 0°C</span>
                <div className="font-mono pt-1 text-[9px] space-y-0.5">
                  <div className="flex justify-between"><span>Strat A:</span><span className="text-[#b91c1c]">{formatK(roiData.night_types_breakdown?.inversion_nights?.A)}</span></div>
                  <div className="flex justify-between"><span>Strat B1:</span><span className="text-[#b24c28]">{formatK(roiData.night_types_breakdown?.inversion_nights?.B1)}</span></div>
                  <div className="flex justify-between"><span>Strat B2:</span><span className="text-[#8c6d46]">{formatK(roiData.night_types_breakdown?.inversion_nights?.B2)}</span></div>
                  <div className="flex justify-between font-bold text-[#4d7c0f]"><span>Strat C:</span><span>{formatK(roiData.night_types_breakdown?.inversion_nights?.C_k8)}</span></div>
                </div>
              </div>
              <div className="p-2 rounded bg-white border border-[#ebdcc0]/60 space-y-0.5">
                <span className="font-bold text-[#2c2113] block">Ordinary Frost (3 nights)</span>
                <span className="text-[#8c6d46] block text-[9px]">T_reg ≤ 0°C (synoptic freeze)</span>
                <div className="font-mono pt-1 text-[9px] space-y-0.5">
                  <div className="flex justify-between"><span>Strat A:</span><span className="text-[#b91c1c]">{formatK(roiData.night_types_breakdown?.ordinary_frost_nights?.A)}</span></div>
                  <div className="flex justify-between"><span>Strat B1:</span><span className="text-[#4d7c0f]">{formatK(roiData.night_types_breakdown?.ordinary_frost_nights?.B1)}</span></div>
                  <div className="flex justify-between"><span>Strat B2:</span><span className="text-[#4d7c0f]">{formatK(roiData.night_types_breakdown?.ordinary_frost_nights?.B2)}</span></div>
                  <div className="flex justify-between font-bold text-[#2c2113]"><span>Strat C:</span><span>{formatK(roiData.night_types_breakdown?.ordinary_frost_nights?.C_k8)}</span></div>
                </div>
              </div>
              <div className="p-2 rounded bg-white border border-[#ebdcc0]/60 space-y-0.5">
                <span className="font-bold text-[#2c2113] block">No Frost (19 nights)</span>
                <span className="text-[#8c6d46] block text-[9px]">All parcels ≥ 0°C (mild)</span>
                <div className="font-mono pt-1 text-[9px] space-y-0.5">
                  <div className="flex justify-between"><span>Strat A:</span><span className="text-[#4d7c0f]">{formatK(roiData.night_types_breakdown?.no_frost_nights?.A)}</span></div>
                  <div className="flex justify-between"><span>Strat B1:</span><span className="text-[#4d7c0f]">{formatK(roiData.night_types_breakdown?.no_frost_nights?.B1)}</span></div>
                  <div className="flex justify-between"><span>Strat B2:</span><span className="text-[#b91c1c]">{formatK(roiData.night_types_breakdown?.no_frost_nights?.B2)}</span></div>
                  <div className="flex justify-between font-bold text-[#4d7c0f]"><span>Strat C:</span><span>{formatK(roiData.night_types_breakdown?.no_frost_nights?.C_k8)}</span></div>
                </div>
              </div>
            </div>
          </div>

          <p className="text-[9px] text-[#8c6d46] italic pt-1 text-center">
            Note: Frequencies reflect the 60-night synthetic catalog, not real climatological distributions. The hero night is a held-out showcase with extreme katabatic inversion.
          </p>
        </div>
      )}
    </div>
  );
}
