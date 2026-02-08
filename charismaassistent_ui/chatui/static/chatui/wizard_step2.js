// wizard_step2.js
(function () {
  // =========================
  // Storage keys
  // =========================
  const TEXT_KEY = "charismaassistant:speechText:v1";
  const COMPARE_KEY = "charismaassistant:enableCompare:v1";

  // =========================
  // Elements
  // =========================
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

  const btnToggleLLM = document.getElementById("btnToggleLLM");
  const elModelBadge = document.getElementById("modelBadge");
  const elRunInfo = document.getElementById("runInfo");

  const btnReanalyze = document.getElementById("btnReanalyze");

  // =========================
  // Utilities
  // =========================
  function safeLocalStorageGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }

  function escapeHtml(str) {
    return (str || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function escapeRegExp(str) {
    return (str || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function splitSentences(text) {
    const raw = (text || "").trim();
    if (!raw) return [];
    const parts = raw
      .replace(/\r\n/g, "\n")
      .split(/(?<=[.!?])\s+(?=[A-ZÄÖÜ„"“])/g);
    return parts.map(s => s.trim()).filter(Boolean);
  }

  function percent(x) {
    const v = Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
    return Math.round(v * 100);
  }

  function setLoading(isLoading, msg) {
    if (btnReanalyze) btnReanalyze.disabled = isLoading;

    // Show lightweight status in runInfo (top right)
    if (elRunInfo) {
      if (isLoading) {
        elRunInfo.textContent = msg || "Analysiere…";
      } else {
        // keep whatever renderSummary puts there; do nothing here
      }
    }

    // Optional: show a simple placeholder in center/left while loading
    if (isLoading) {
      if (elTbody) elTbody.innerHTML = `<tr><td colspan="5" class="muted">Analysiere…</td></tr>`;
      if (elSpeechPanel) {
        elSpeechPanel.innerHTML = `<div class="muted">Analysiere…</div>`;
      }
      renderDetails(null);
    }
  }

  async function fetchAnalysisFromBackend(text, enableCompare) {
    const resp = await fetch("/api/analyze/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Django proxy expects: {text, enable_compare}
      body: JSON.stringify({ text, enable_compare: !!enableCompare }),
    });

    if (!resp.ok) {
      let detail = "";
      try { detail = await resp.text(); } catch {}
      throw new Error(`Backend error ${resp.status}: ${detail || resp.statusText}`);
    }

    return await resp.json(); // expected: { analyses: { gpt: ..., gemini?: ... } }
  }

  // =========================
  // Rendering
  // =========================
  function renderSummary(analysis) {
    if (!analysis) return;

    if (elSummaryScore) elSummaryScore.textContent = `${analysis.overall_score} / 100`;

    const items = Array.isArray(analysis.items) ? analysis.items : [];
    const found = items.filter(x => x && x.present).length;
    if (elSummaryFound) elSummaryFound.textContent = `${found} von ${items.length}`;

    const cons = analysis.meta?.consistency;
    if (elSummaryConsistency) {
      elSummaryConsistency.textContent =
        (typeof cons === "number" && Number.isFinite(cons)) ? cons.toFixed(2) : "—";
    }

    const st = analysis.meta?.setup;
    if (elSummarySetup) {
      elSummarySetup.textContent = st
        ? `${st.clts} verbale CLTs · ${st.llm_runs} LLM-Runs · ${st.mode}`
        : "—";
    }

    // runInfo: show actual model name if present (optional)
    const modeStr = analysis.meta?.setup?.mode || "";
    if (elRunInfo) elRunInfo.textContent = modeStr ? `Model: ${modeStr}` : "—";
  }

  function renderStrengthPips(strength, max = 3) {
    const p = [];
    for (let i = 1; i <= max; i++) {
      p.push(`<span class="pip ${i <= strength ? "on" : "off"}"></span>`);
    }
    return `<span class="pips">${p.join("")}</span> <span class="muted">${strength}/${max}</span>`;
  }

  function renderTable(analysis, onSelect) {
    if (!elTbody) return;
    elTbody.innerHTML = "";

    const items = Array.isArray(analysis.items) ? analysis.items : [];
    items.forEach((it) => {
      const tr = document.createElement("tr");
      tr.dataset.cltId = it.clt_id;

      const check = it.present ? "✓" : "×";
      const confPct = percent(it.confidence);

      tr.innerHTML = `
        <td>${escapeHtml(it.label)}</td>
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
    if (!elSpeechPanel) return;

    const sentences = splitSentences(text);

    // Evidence map: clt_id -> quotes[]
    const evidMap = new Map();
    const items = Array.isArray(analysis.items) ? analysis.items : [];
    items.forEach(it => {
      const quotes = (it.evidence || [])
        .map(e => (e && e.quote ? String(e.quote).trim() : ""))
        .filter(Boolean);
      evidMap.set(it.clt_id, quotes);
    });

    elSpeechPanel.innerHTML = "";

    sentences.forEach((s, idx) => {
      let html = escapeHtml(s);

      items.forEach(it => {
        const quotes = evidMap.get(it.clt_id) || [];
        quotes.forEach(qRaw => {
          const q = String(qRaw || "").trim();
          if (!q) return;

          const escapedQ = escapeHtml(q);
          if (!html.includes(escapedQ)) return;

          const cls = it.ui_color ? `hl ${it.ui_color}` : "hl";
          const re = new RegExp(escapeRegExp(escapedQ), "g");
          html = html.replace(re, `<span class="${cls}" data-clt="${it.clt_id}">${escapedQ}</span>`);
        });
      });

      const div = document.createElement("div");
      div.className = "sentence";
      div.innerHTML = `<small>Satz ${idx + 1}</small>${html}`;

      if (selectedCltId && div.querySelector(`[data-clt="${selectedCltId}"]`)) {
        div.style.borderColor = "rgba(106,167,255,35)";
        div.style.background = "rgba(106,167,255,05)";
      }

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
    if (!elDetailPanel || !elDetailTitle || !elDetailMeta || !elDetailBody) return;

    if (!item) {
      elDetailPanel.dataset.open = "false";
      elDetailTitle.textContent = "Details";
      elDetailMeta.textContent = "Wähle eine CLT aus.";
      elDetailBody.innerHTML = `<div class="muted">Noch keine Auswahl.</div>`;
      return;
    }

    elDetailPanel.dataset.open = "true";
    elDetailTitle.textContent = item.label;
    elDetailMeta.textContent = `Stärke: ${item.strength}/3 · Vertrauen: ${Number(item.confidence || 0).toFixed(2)}`;

    const ev = (item.evidence || []).map(e => e?.quote).filter(Boolean);
    const evHtml = ev.length
      ? `<ul>${ev.map(q => `<li><span class="badge">Evidenz</span> <span class="muted">„${escapeHtml(q)}“</span></li>`).join("")}</ul>`
      : `<div class="muted">Keine Evidenz (kein Treffer).</div>`;

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

  // =========================
  // State
  // =========================
  let STATE = {
    text: "",
    activeModel: "gpt",
    analyses: { gpt: null, gemini: null },
    selected: null,
  };

  function updateSelectedRow(cltId) {
    document.querySelectorAll("#scoreTbody tr").forEach(tr => {
      tr.classList.toggle("is-selected", tr.dataset.cltId === cltId);
    });
  }

  function getActiveAnalysis() {
    return STATE.analyses[STATE.activeModel];
  }

  function renderAll() {
    const analysis = getActiveAnalysis();
    if (!analysis) return;

    renderSummary(analysis);
    renderTable(analysis, (id) => selectCLT(id));
    renderSpeechPanel(STATE.text, analysis, STATE.selected);
    renderDetails(null);
  }

  function selectCLT(cltId) {
    const analysis = getActiveAnalysis();
    if (!analysis || !Array.isArray(analysis.items)) return;

    STATE.selected = cltId;
    updateSelectedRow(cltId);

    const item = analysis.items.find(x => x.clt_id === cltId);
    renderDetails(item || null);
    renderSpeechPanel(STATE.text, analysis, cltId);
  }

  window.selectCLT = selectCLT;

  function setActiveModel(model) {
    if (!STATE.analyses[model]) return;

    STATE.activeModel = model;

    if (elModelBadge) elModelBadge.textContent = model.toUpperCase();

    if (btnToggleLLM) {
      btnToggleLLM.dataset.active = model;
      btnToggleLLM.textContent = (model === "gpt")
        ? "LLM auswählen: Gemini"
        : "LLM auswählen: GPT";
    }

    STATE.selected = null;
    updateSelectedRow(null);

    renderAll();
  }

  function getCompareFlag() {
    return safeLocalStorageGet(COMPARE_KEY) === "1";
  }

  function showError(err) {
    const msg = (err && err.message) ? err.message : String(err);

    if (elSummaryScore) elSummaryScore.textContent = "—";
    if (elSummaryFound) elSummaryFound.textContent = "—";
    if (elSummaryConsistency) elSummaryConsistency.textContent = "—";
    if (elSummarySetup) elSummarySetup.textContent = "Backend-Fehler";

    if (elRunInfo) elRunInfo.textContent = "—";

    if (elTbody) {
      elTbody.innerHTML = `<tr><td colspan="5" class="muted">Analyse fehlgeschlagen: ${escapeHtml(msg)}</td></tr>`;
    }

    if (elSpeechPanel) {
      elSpeechPanel.innerHTML = `<div class="muted">Analyse fehlgeschlagen: ${escapeHtml(msg)}</div>`;
    }

    if (btnToggleLLM) btnToggleLLM.style.display = "none";
    renderDetails(null);
  }

  async function runAnalysis() {
    const text = safeLocalStorageGet(TEXT_KEY) || "";
    STATE.text = text;

    const enableCompare = getCompareFlag();

    // Optional: if you inject results server-side later, we still accept them
    // but backend call is the default path.
    const injectedGpt = window.RESULT_GPT || null;
    const injectedGemini = window.RESULT_GEMINI || null;

    setLoading(true, "Analysiere…");

    try {
      // If server injected results are present, use them immediately
      if (injectedGpt) {
        STATE.analyses.gpt = injectedGpt;
        STATE.analyses.gemini = injectedGemini || null;
      } else {
        const data = await fetchAnalysisFromBackend(text, enableCompare);
        STATE.analyses.gpt = data?.analyses?.gpt || null;
        STATE.analyses.gemini = data?.analyses?.gemini || null;
      }

      // Toggle button visibility
      if (!STATE.analyses.gemini) {
        if (btnToggleLLM) btnToggleLLM.style.display = "none";
      } else {
        if (btnToggleLLM) btnToggleLLM.style.display = "";
      }

      // Default model
      setActiveModel("gpt");
      renderDetails(null);

      // Enable reanalyze now that first run succeeded
      if (btnReanalyze) btnReanalyze.disabled = false;

      // Attach toggle handler once
      if (btnToggleLLM && STATE.analyses.gemini && !btnToggleLLM.dataset.bound) {
        btnToggleLLM.dataset.bound = "1";
        btnToggleLLM.addEventListener("click", () => {
          const next = (STATE.activeModel === "gpt") ? "gemini" : "gpt";
          setActiveModel(next);
        });
      }
    } catch (err) {
      console.error(err);
      showError(err);
    } finally {
      setLoading(false);
    }
  }

  // =========================
  // Events
  // =========================
  if (btnCloseDetails) {
    btnCloseDetails.addEventListener("click", () => {
      STATE.selected = null;
      updateSelectedRow(null);
      renderDetails(null);
      const analysis = getActiveAnalysis();
      if (analysis) renderSpeechPanel(STATE.text, analysis, null);
    });
  }

  if (btnReanalyze) {
    btnReanalyze.addEventListener("click", async () => {
      await runAnalysis();
    });
  }

  // =========================
  // Init
  // =========================
  // Keep default disabled until first successful run
  if (btnReanalyze) btnReanalyze.disabled = true;

  runAnalysis();
})();
