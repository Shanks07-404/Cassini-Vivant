# Cassini Vivant

> *"La triangulation du terroir, en temps réel"*  
> **MTC ThinkAI'26 Prototype — Viticulture Microclimate AI & Appellation Adaptation**

---

## 1. Quick Start

Cassini Vivant is designed to run **100% offline** with zero downloads and zero external API keys. The optional Gemini assistant falls back gracefully to deterministic cached answers when no API key is supplied.

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm

### Running the Backend
From the repository root (`cassini-vivant`):
```bash
# 1. Activate your virtual environment (or let uv run it)
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate

# 2. Run backend API server
uvicorn backend.app.main:app --reload --port 8000
```
API Documentation will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Running the Frontend
From the `cassini-vivant/frontend` directory:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### Running Tests
From the repository root (`cassini-vivant`):
```bash
# Run backend pytest suite (all unit & API tests)
pytest backend/tests -v

# Run frontend build verification
cd frontend && npm run build
```

### Verifying Zero External Network Requests
The frontend is 100% offline-ready:
1. Open DevTools in Chrome / Edge / Firefox (`F12` or `Ctrl+Shift+I`).
2. Go to the **Network** tab.
3. Refresh the page [http://localhost:5173](http://localhost:5173).
4. Verify all requests originate from `localhost:5173` or `127.0.0.1:8000`. No Google Fonts, unpkg, cdnjs, or remote map tile requests are made.
5. In the Network tab dropdown, select **Offline** and refresh: the entire map, canvas choropleth, UI controls, and replay scrubbers continue to function completely.

---

## 2. Configuration & Environment Variables

Copy `.env.example` to `.env` if not already present:
```bash
cp .env.example .env
```
Key variables:
- `GEMINI_API_KEY`: *(Optional)* Your Google AI Studio API key. If omitted, the assistant operates in offline demo mode using precomputed answers.
- `GEMINI_MODEL`: *(Optional)* Gemini model identifier (e.g. `gemini-2.5-flash`). Do not hardcode this in source files.
- `DEFAULT_LANGUAGE`: `en` or `fr`.

---

## 3. Scientific & Algorithmic Foundation

Inspired by César-François Cassini de Thury's 1747 geodetic map of France, Cassini Vivant measures a small number of reference points (weather stations) and derives microclimatic conditions mathematically for every parcel:

- **Hyperparameter Calibration & Survey Implications:**
  - *Dense Historical Reference Network:* Fitting across training nights 002–060 on a dense reference mesh estimates vertical length-scale $\ell_h \approx 15.4\text{ m}$ (95% CI $[12.4, 18.4]\text{ m}$), capturing the steep nocturnal inversion layer. This requires a one-time high-density mobile sensor campaign.
  - *On-Station Calibration (6 fixed slope/ridge stations):* Fitting via leave-one-out cross-validation yields a much flatter vertical lapse scale $\ell_h \approx 148.9\text{ m}$ ($\text{RMSE} = 0.975^\circ\text{C}$ on slope stations). It costs €0 in prior survey expenses but underestimates low-elevation katabatic pooling prior to active sensor placement.
- **Cost-Loss Decision Theory & Strategy Comparison:**
  - Optimal intervention rule: $p^* = \text{cost} / \text{loss} = 350 / 4500 \approx 7.8\%$. Break-even ratio: $\text{loss} / \text{cost} = 12.86\times$.
  - **Strategy A (Protect Nothing):** €0 protection, unmitigated crop loss.
  - **Strategy B1 (Regional-Alert Rule, $T_{\text{reg}} \le 0^\circ\text{C}$):** On the hero night, regional forecast minimum is $+2.4^\circ\text{C}$, so regional-alert systems protect 0 parcels and suffer the same €220,500 damage as doing nothing.
  - **Strategy B2 (Blanket Protection):** Protect all 239 parcels every night (cost upper bound: €83,650/night).
  - **Strategy C (Cassini Vivant):** Targeted protection when $P(\text{frost}) \ge p^*$. With default 6 stations ($K=6$), expenditure is €165,900 on the hero night; adding 2 decision-focused valley stations ($K=8$) reduces expenditure to €90,600 (and €35,005 on typical training nights).
- **Optimal Sensor Placement:** Greedy decision-focused active learning ($\arg\max_i \Delta \text{DecisionScore}$) targeting parcels near the $0^\circ\text{C}$ transition boundary.
- **Appellation Adaptation Dossier:** Automated synthesis of multi-night microclimate exceedance evidence under INAO/ODG regulatory framework with human review guardrails.

---

## 4. Honesty & Ethical Guardrails

- **SIMULATED DATA Badge:** Prominently displayed across the map and dashboard.
- **DEMO DATA Watermark:** Prominently watermarked across exported appellation evidence dossiers.
- **ILLUSTRATIVE ROI:** All economic figures (candle costs, turbine costs, grape yield value) are labeled as illustrative baselines.
- **SIMPLIFIED DEMO VERSION:** All appellation rulebooks in `backend/rules/` are simplified demonstrations and clearly state that regulatory changes require INAO review.
- **Fact Verification:** All historical and market claims in `content/story.json` are explicitly marked with `"needs_verification": true`.
