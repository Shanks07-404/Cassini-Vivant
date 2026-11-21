import { useState } from 'react';
import { Send, FileText, ExternalLink, ShieldCheck, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { Language, getTranslation } from '../i18n';
import { ParcelProperties } from '../types';

interface AssistantTabProps {
  lang: Language;
  selectedParcel?: ParcelProperties | null;
}

interface Citation {
  doc_id: string;
  section: string;
  title: string;
}

interface RuleExcerpt {
  doc_id: string;
  section_title: string;
  content: string;
  citation: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  rule_excerpts?: RuleExcerpt[];
  timestamp: string;
}

const DEMO_PROMPTS = [
  {
    label: "Hero Night Diagnostic",
    query: "What happened on the hero night night_001 and why did regional alerts fail?",
  },
  {
    label: "ROI of Adding 2 Valley Sensors (K=8)",
    query: "What is the ROI and economic benefit of adding decision-focused sensors (K=8 vs K=6)?",
  },
  {
    label: "Aspersion & Bougie Regulations",
    query: "What are the regulatory and environmental limits on water aspersion and bougies?",
  },
  {
    label: "Compare Strategy B1, B2 & Cassini C",
    query: "How does Cassini Vivant compare against benchmark strategies B1, B2, B3, and B4?",
  },
];

export default function AssistantTab({ lang, selectedParcel }: AssistantTabProps) {
  const t = getTranslation(lang);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: lang === 'fr'
        ? "Bienvenue sur l'Assistant ODG Cassini Vivant. Je suis synchronisé avec le cahier des charges de l'AOP, les règles environnementales INAO/DREAL et les données calibrées de microclimat. Comment puis-je vous guider dans votre arbitrage antigel ?"
        : "Welcome to the Cassini Vivant ODG Appellation Assistant. I am equipped with the AOP specifications, INAO/DREAL environmental regulations, and calibrated microclimate evaluation data. How may I assist your frost management strategy?",
      timestamp: "04:00",
      citations: [
        { doc_id: "AOP-CHAB-2024", section: "1.1", title: "Moyens Actifs Autorisés" },
        { doc_id: "CAS-SOP-FROST", section: "3.1", title: "Règle Coût-Perte p*" },
      ],
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedExcerpts, setExpandedExcerpts] = useState<number | null>(null);

  const sendMessage = async (textToSend?: string) => {
    const text = (textToSend || input).trim();
    if (!text || loading) return;

    const userMsg: Message = {
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          parcel_id: selectedParcel?.id,
          lang,
        }),
      });

      if (!res.ok) throw new Error('API error');
      const data = await res.json();

      const assistantMsg: Message = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        rule_excerpts: data.rule_excerpts || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Chat request failed:', err);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: lang === 'fr'
            ? "Une erreur de communication est survenue. Veuillez vérifier votre connexion au serveur local."
            : "A communication error occurred. Please verify your connection to the local backend.",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const openDossier = () => {
    window.open(`/api/dossier/html?lang=${lang}`, '_blank');
  };

  return (
    <div className="flex flex-col h-[650px] font-serif text-xs text-[#2c2113]">
      {/* Header & AI Disclosure Banner */}
      <div className="space-y-2 mb-3">
        <div className="p-3 rounded-lg border border-[#8c6d46]/40 bg-[#f5efe0] space-y-1">
          <div className="flex items-center justify-between">
            <span className="font-sans font-bold uppercase tracking-wider text-[11px] text-[#4a3821] flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-[#4d7c0f]" />
              <span>{t.assistant.title}</span>
            </span>
            <span className="text-[10px] font-mono text-[#8c6d46] bg-[#ebdcc0]/60 px-2 py-0.5 rounded">
              INAO / AOP Connected
            </span>
          </div>
          <p className="text-[11px] text-[#6b5233] leading-relaxed">
            {t.assistant.banner}
          </p>
          <div className="text-[10px] text-[#8c6d46] italic pt-1 flex items-center gap-1">
            <Info className="w-3 h-3 shrink-0" />
            <span>{t.assistant.legal_notice}</span>
          </div>
        </div>

        {/* Selected Parcel Badge if any */}
        {selectedParcel && (
          <div className="flex items-center justify-between px-3 py-1.5 bg-[#ebdcc0]/40 border border-[#ebdcc0] rounded text-[11px] font-sans">
            <span className="text-[#6b5233]">
              Active Parcel Context: <strong>{selectedParcel.id} ({selectedParcel.name})</strong>
            </span>
            <span className="text-[10px] font-mono text-[#8c6d46]">
              Elev: {selectedParcel.elevation.toFixed(1)}m | Rel: {selectedParcel.rel_elev.toFixed(1)}m
            </span>
          </div>
        )}

        {/* Quick Demo Prompts Chips */}
        <div className="space-y-1">
          <span className="font-sans font-bold text-[10px] uppercase text-[#8c6d46]">
            {t.assistant.quick_prompts}
          </span>
          <div className="flex flex-wrap gap-1.5">
            {DEMO_PROMPTS.map((dp, i) => (
              <button
                key={i}
                onClick={() => sendMessage(dp.query)}
                disabled={loading}
                className="text-[10px] font-sans bg-[#faf7f0] hover:bg-[#ebdcc0] text-[#4a3821] border border-[#ebdcc0] px-2.5 py-1 rounded-full transition-colors disabled:opacity-50 text-left"
              >
                ✦ {dp.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chat Messages Log */}
      <div className="flex-1 overflow-y-auto space-y-3 p-3 bg-[#faf7f0] border border-[#ebdcc0] rounded-lg shadow-inner">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[88%] p-3 rounded-lg text-[12px] leading-relaxed ${
                m.role === 'user'
                  ? 'bg-[#8c6d46] text-[#faf7f0] rounded-br-none shadow-sm'
                  : 'bg-[#ffffff] text-[#2c2113] border border-[#ebdcc0] rounded-bl-none shadow-sm'
              }`}
            >
              {/* Message text with basic newline formatting */}
              <div className="whitespace-pre-line font-serif">
                {m.content}
              </div>

              {/* Citations Pills */}
              {m.citations && m.citations.length > 0 && (
                <div className="mt-2.5 pt-2 border-t border-[#ebdcc0]/60 space-y-1">
                  <span className="font-sans font-bold text-[9px] uppercase tracking-wider text-[#8c6d46] block">
                    Authoritative Regulatory Citations:
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {m.citations.map((c, ci) => (
                      <span
                        key={ci}
                        className="font-mono text-[9px] bg-[#f5efe0] text-[#8e391b] border border-[#d4a373]/60 px-1.5 py-0.5 rounded font-bold"
                      >
                        [{c.doc_id} § {c.section}]
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Expandable Rule Excerpts */}
              {m.rule_excerpts && m.rule_excerpts.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => setExpandedExcerpts(expandedExcerpts === idx ? null : idx)}
                    className="font-sans text-[10px] text-[#8c6d46] hover:text-[#4a3821] flex items-center gap-1 font-bold underline"
                  >
                    <span>{expandedExcerpts === idx ? 'Hide Regulatory Details' : 'View Rulebook Excerpts'}</span>
                    {expandedExcerpts === idx ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>

                  {expandedExcerpts === idx && (
                    <div className="mt-2 space-y-2 font-sans bg-[#f7f2e7] p-2 rounded border border-[#ebdcc0] text-[10px]">
                      {m.rule_excerpts.map((rx, ri) => (
                        <div key={ri} className="border-b border-[#ebdcc0]/60 pb-1.5 last:border-b-0">
                          <div className="font-bold text-[#8e391b]">{rx.citation}</div>
                          <div className="text-[#4a3821] mt-0.5 italic">{rx.content}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            <span className="font-mono text-[9px] text-[#8c6d46] mt-0.5 px-1">
              {m.timestamp}
            </span>
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-xs italic text-[#8c6d46] p-2">
            <span className="animate-spin">✦</span> Consulting appellation rulebooks and microclimate models...
          </div>
        )}
      </div>

      {/* Input bar and Dossier export button */}
      <div className="mt-3 space-y-2">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t.assistant.input_placeholder}
            disabled={loading}
            className="flex-1 px-3 py-2 text-xs rounded border border-[#ebdcc0] bg-white focus:outline-none focus:border-[#8c6d46] font-sans"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-[#8c6d46] hover:bg-[#6b5233] text-[#faf7f0] rounded font-sans font-bold text-xs flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <Send className="w-3.5 h-3.5" />
            <span>{t.assistant.send}</span>
          </button>
        </form>

        {/* Dossier Download Action */}
        <button
          onClick={openDossier}
          className="w-full py-2 px-3 border border-[#8c6d46]/50 bg-[#f5efe0] hover:bg-[#ebdcc0] text-[#4a3821] rounded text-[11px] font-sans font-bold flex items-center justify-center gap-2 transition-colors shadow-sm"
        >
          <FileText className="w-3.5 h-3.5 text-[#8c6d46]" />
          <span>{t.assistant.download_dossier}</span>
          <ExternalLink className="w-3 h-3 text-[#8c6d46]" />
        </button>
      </div>
    </div>
  );
}
