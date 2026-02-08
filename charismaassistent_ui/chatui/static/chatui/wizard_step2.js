(function () {
  // Keep consistent with Step 1 storage key
  const TEXT_KEY = "charismaassistant:speechText:v1";
  const MOCK_KEY = "charismaassistant:mockAnalysis:v1";

  const elSpeechPanel = document.getElementById("speechPanel");
  const elTbody = document.getElementById("scoreTbody");
  const elDetailPanel = document.getElementById("detailPanel");
  const elDetailTitle = document.getElementById("detailTitle");
  const elDetailMeta = document.getElementById("detailMeta");
  const elDetailBody = document.getElementById("detailBody");

  const elSummaryScore = document.getElementById("summaryScore");
  const elSummaryFound = document.getElementById("summaryFound");
  const elSummaryConsistency = document.getElementById("summaryConsistency");
  const elSummarySetup = document.getElementById("summarySetup");

  const btnCloseDetails = document.getElementById("btnCloseDetails");

  // --- Utilities ---
  function safeLocalStorageGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }
  function safeLocalStorageSet(key, value) {
    try { localStorage.setItem(key, value); } catch {}
  }

  function splitSentences(text) {
    // Pragmatic sentence split for German/English speeches
    // Keeps punctuation; good enough for UI highlight.
    const raw = (text || "").trim();
    if (!raw) return [];
    const parts = raw
      .replace(/\r\n/g, "\n")
      .split(/(?<=[.!?])\s+(?=[A-ZÄÖÜ„"“])/g);
    return parts.map(s => s.trim()).filter(Boolean);
  }

  function percent(x) {
    const v = Math.max(0, Math.min(1, x || 0));
    return Math.round(v * 100);
  }

  // --- Mock analysis (until FastAPI/LLM wired) ---
  function buildMockAnalysis(text) {
    // Deterministic-ish mock: use length to vary
    const base = Math.min(0.95, 0.55 + (text.length % 200) / 500);
    const clts = [
      { key: "moral_conviction", name: "Moral Conviction", color: "clt-moral" },
      { key: "collective_sentiment", name: "Collective Sentiment", color: "clt-collective" },
      { key: "lists_repetition", name: "Lists / Repetition", color: "" },
      { key: "rhetorical_question", name: "Rhetorical Question", color: "clt-rhetq" },
      { key: "contrast", name: "Contrast", color: "clt-contrast" },
      { key: "story_anecdote", name: "Story / Anecdote", color: "" },
      { key: "metaphor_simile", name: "Metaphor / Simile", color: "" },
      { key: "ambitious_goals", name: "Ambitious Goals", color: "" },
      { key: "confidence_in_goals", name: "Confidence in Goals", color: "" },
    ];

    const sents = splitSentences(text);
    const pickQuote = (idx) => (sents[idx] ? sents[idx].slice(0, 90) : "");

    // Simple fake evidence from first few sentences
    const items = clts.map((c, i) => {
      const strength = (i % 4); // 0..3
      const present = strength >= 1;
      const conf = Math.max(0.50, Math.min(0.95, base - i * 0.03));
      const evidences = present && sents.length
        ? [{ quote: pickQuote(Math.min(i, sents.length - 1)) }]
        : [];
      return {
        clt_id: c.key,
        label: c.name,
        present,
        strength,
        confidence: conf,
        evidence: evidences,
        rationale_short: present
          ? "Mock-Begründung (LLM noch nicht verbunden)."
          : "Mock: keine klaren Hinweise gefunden.",
        ui_color: c.color
      };
    });

    const found = items.filter(x => x.present).length;
    const overall = Math.round(
      (items.reduce((acc, x) => acc + (x.strength / 3) * x.confidence, 0) / items.length) * 100
    );

    return {
      meta: {
        setup: { clts: 9, llm_runs: 3, mode: "Rule+LLM (Mock)" },
        consistency: 0.78,
      },
      overall_score: overall,
      items
    };
  }

  // --- Rendering ---
  function renderSummary(analysis) {
    elSummaryScore.textContent = `${analysis.overall_score} / 100`;
    const found = analysis.items.filter(x => x.present).length;
    elSummaryFound.textContent = `${found} von ${analysis.items.length}`;
    elSummaryConsistency.textContent = (analysis.meta?.consistency ?? "—").toFixed
      ? (analysis.meta.consistency).toFixed(2)
      : `${analysis.meta?.consistency ?? "—"}`;
    const st = analysis.meta?.setup;
    elSummarySetup.textContent = st
      ? `${st.clts} verbale CLTs · ${st.llm_runs} LLM-Runs · ${st.mode}`
      : "—";
  }

  function renderStrengthPips(strength, max = 3) {
    const p = [];
    for (let i = 1; i <= max; i++) {
      p.push(`<span class="pip ${i <= strength ? "on" : "off"}"></span>`);
    }
    return `<span class="pips">${p.join("")}</span> <span class="muted">${strength}/${max}</span>`;
  }

  function renderTable(analysis, onSelect) {
    elTbody.innerHTML = "";
    analysis.items.forEach((it) => {
      const tr = document.createElement("tr");
      tr.dataset.cltId = it.clt_id;

      const check = it.present ? "✓" : "×";
      const confPct = percent(it.confidence);

      tr.innerHTML = `
        <td>${it.label}</td>
        <td style="font-weight:800;">${check}</td>
        <td><div class="strength">${renderStrengthPips(it.strength, 3)}</div></td>
        <td style="font-weight:700;">${confPct} %</td>
        <td><span class="badge">Öffnen</span></td>
      `;

      tr.addEventListener("click", () => onSelect(it.clt_id));
      elTbody.appendChild(tr);
    });
  }

  function renderSpeechPanel(text, analysis, selectedCltId) {
    const sentences = splitSentences(text);

    // Build evidence map: clt_id -> array of quotes
    const evidMap = new Map();
    analysis.items.forEach(it => {
      evidMap.set(it.clt_id, (it.evidence || []).map(e => (e.quote || "").trim()).filter(Boolean));
    });

    elSpeechPanel.innerHTML = "";

    sentences.forEach((s, idx) => {
      let html = escapeHtml(s);

      // Highlight evidence quotes inside this sentence (simple contains)
      analysis.items.forEach(it => {
        const quotes = evidMap.get(it.clt_id) || [];
        quotes.forEach(q => {
          if (!q) return;
          // naive match: if quote fragment is inside sentence
          if (s.includes(q)) {
            const cls = it.ui_color ? `hl ${it.ui_color}` : "hl";
            html = html.replaceAll(escapeHtml(q), `<span class="${cls}" data-clt="${it.clt_id}">${escapeHtml(q)}</span>`);
          }
        });
      });

      const div = document.createElement("div");
      div.className = "sentence";
      div.innerHTML = `<small>Satz ${idx + 1}</small>${html}`;

      // If sentence contains any highlight for selected CLT, add subtle emphasis
      if (selectedCltId && div.querySelector(`[data-clt="${selectedCltId}"]`)) {
        div.style.borderColor = "rgba(106,167,255,.35)";
        div.style.background = "rgba(106,167,255,.05)";
      }

      // Clicking a highlight selects that CLT
      div.addEventListener("click", (ev) => {
        const target = ev.target;
        if (target && target.dataset && target.dataset.clt) {
          selectCLT(target.dataset.clt);
          ev.stopPropagation();
        }
      });

      elSpeechPanel.appendChild(div);
    });

    if (!sentences.length) {
      elSpeechPanel.innerHTML = `<div class="muted">Kein Text gefunden (Step 1).</div>`;
    }
  }

  function renderDetails(item) {
    if (!item) {
      elDetailPanel.dataset.open = "false";
      elDetailTitle.textContent = "Details";
      elDetailMeta.textContent = "Wähle eine CLT aus.";
      elDetailBody.innerHTML = `<div class="muted">Noch keine Auswahl.</div>`;
      return;
    }

    elDetailPanel.dataset.open = "true";
    elDetailTitle.textContent = item.label;
    elDetailMeta.textContent = `Stärke: ${item.strength}/3 · Vertrauen: ${item.confidence.toFixed(2)}`;

    const ev = (item.evidence || []).map(e => e.quote).filter(Boolean);
    const evHtml = ev.length
      ? `<ul>${ev.map(q => `<li><span class="badge">Evidenz</span> <span class="muted">„${escapeHtml(q)}“</span></li>`).join("")}</ul>`
      : `<div class="muted">Keine Evidenz (Mock/kein Treffer).</div>`;

    // Simple accordion blocks (no extra CSS needed)
    elDetailBody.innerHTML = `
      <details open class="card" style="padding:10px; margin-bottom:10px;">
        <summary style="cursor:pointer; font-weight:700;">Evidenz</summary>
        <div style="margin-top:8px;">${evHtml}</div>
      </details>

      <details open class="card" style="padding:10px; margin-bottom:10px;">
        <summary style="cursor:pointer; font-weight:700;">Begründung</summary>
        <div class="muted" style="margin-top:8px;">${escapeHtml(item.rationale_short || "—")}</div>
      </details>

      <details class="card" style="padding:10px;">
        <summary style="cursor:pointer; font-weight:700;">Debug (optional)</summary>
        <pre class="result__text" style="margin-top:8px; max-height:240px; overflow:auto;">${escapeHtml(JSON.stringify(item, null, 2))}</pre>
      </details>
    `;
  }

  function escapeHtml(str) {
    return (str || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  // --- Selection handling ---
  let STATE = { analysis: null, text: "", selected: null };

  function updateSelectedRow(cltId) {
    document.querySelectorAll("#scoreTbody tr").forEach(tr => {
      tr.classList.toggle("is-selected", tr.dataset.cltId === cltId);
    });
  }

  function selectCLT(cltId) {
    STATE.selected = cltId;
    updateSelectedRow(cltId);

    const item = STATE.analysis.items.find(x => x.clt_id === cltId);
    renderDetails(item);
    renderSpeechPanel(STATE.text, STATE.analysis, cltId);
  }

  // Expose for highlight click
  window.selectCLT = selectCLT;

  // --- Init ---
  function init() {
    const text = safeLocalStorageGet(TEXT_KEY) || "";
    STATE.text = text;

    // Use stored mock analysis if available (so UI stays stable)
    const stored = safeLocalStorageGet(MOCK_KEY);
    let analysis = null;
    if (stored) {
      try { analysis = JSON.parse(stored); } catch {}
    }
    if (!analysis) {
      analysis = buildMockAnalysis(text);
      safeLocalStorageSet(MOCK_KEY, JSON.stringify(analysis));
    }

    STATE.analysis = analysis;

    renderSummary(analysis);
    renderTable(analysis, (id) => selectCLT(id));
    renderSpeechPanel(text, analysis, null);
    renderDetails(null);
  }

  btnCloseDetails.addEventListener("click", () => {
    STATE.selected = null;
    updateSelectedRow(null);
    renderDetails(null);
    renderSpeechPanel(STATE.text, STATE.analysis, null);
  });

  init();
})();
