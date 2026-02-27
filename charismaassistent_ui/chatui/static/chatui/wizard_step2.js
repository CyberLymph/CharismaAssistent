// wizard_step2.js (EN + HYBRID optional + streaming NDJSON)
(function () {
  const TEXT_KEY = "charismaassistant:speechText:v1";
  const COMPARE_KEY = "charismaassistant:enableCompare:v1";
  const HYBRID_KEY = "charismaassistant:enableHybrid:v1";

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

  const elRunDistributionWrap = document.getElementById("runDistributionWrap");
  const elRunDistribution = document.getElementById("runDistribution");

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

  function nowTime() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function getCompareFlag() {
    return safeLocalStorageGet(COMPARE_KEY) === "1";
  }

  function getHybridFlag() {
    return safeLocalStorageGet(HYBRID_KEY) === "1";
  }

  // --- Run log ---
  function runLogReset() {
    if (!elRunDistribution) return;
    elRunDistribution.innerHTML = "";
  }

  function runLogAppend(line) {
    if (!elRunDistribution) return;
    const div = document.createElement("div");
    div.className = "muted";
    div.style.padding = "2px 0";
    div.textContent = line;
    elRunDistribution.appendChild(div);
    elRunDistribution.scrollTop = elRunDistribution.scrollHeight;
  }

  function setRunBoxVisible(visible) {
    if (!elRunDistributionWrap) return;
    elRunDistributionWrap.style.display = visible ? "" : "none";
    if (visible) {
      // Ensure we have some initial content so it's not "empty"
      if (elRunDistribution && elRunDistribution.children.length === 0) {
        runLogAppend(`[${nowTime()}] Waiting for runs…`);
      }
    }
  }

  function setLoading(isLoading, msg) {
    if (btnReanalyze) btnReanalyze.disabled = isLoading;

    if (elRunInfo && isLoading) {
      elRunInfo.textContent = msg || "Analyzing…";
    }

    if (isLoading) {
      if (elTbody) elTbody.innerHTML = `<tr><td colspan="5" class="muted">Analyzing…</td></tr>`;
      if (elSpeechPanel) elSpeechPanel.innerHTML = `<div class="muted">Analyzing…</div>`;
      renderDetails(null);

      // If run box visible, reset it
      if (getHybridFlag()) {
        runLogReset();
        runLogAppend(`[${nowTime()}] Starting hybrid analysis…`);
      }
    }
  }

  // =========================
  // Backend calls
  // =========================
  async function fetchAnalysisFromBackend(text, enableCompare, useHybrid) {
    const resp = await fetch("/api/analyze/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        enable_compare: !!enableCompare,
        use_hybrid: !!useHybrid
      }),
    });

    if (!resp.ok) {
      let detail = "";
      try { detail = await resp.text(); } catch {}
      throw new Error(`Backend error ${resp.status}: ${detail || resp.statusText}`);
    }

    return await resp.json(); // { analyses: { gpt:..., gemini?:... } }
  }

  async function streamAnalysisFromBackend(text, enableCompare, useHybrid) {
    const resp = await fetch("/api/analyze_stream/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        enable_compare: !!enableCompare,
        use_hybrid: !!useHybrid
      }),
    });

    if (!resp.ok) {
      let detail = "";
      try { detail = await resp.text(); } catch {}
      throw new Error(`Backend error ${resp.status}: ${detail || resp.statusText}`);
    }

    if (!resp.body) throw new Error("Streaming not supported by this browser/response.");

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalPayload = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);

        if (!line) continue;

        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          runLogAppend(`[${nowTime()}] (warning) Non-JSON: ${line.slice(0, 120)}`);
          continue;
        }

        if (msg.type === "run_done") {
          const model = (msg.model || "gpt").toUpperCase();
          const run = Number(msg.run || 0);
          const score = (typeof msg.overall_score === "number") ? msg.overall_score : "—";
          runLogAppend(`[${nowTime()}] Run ${run} done (${model}) · score: ${score}/100`);
        } else if (msg.type === "progress") {
          runLogAppend(`[${nowTime()}] ${msg.message || "Progress…"}`);
        } else if (msg.type === "final") {
          finalPayload = msg;
          runLogAppend(`[${nowTime()}] Final aggregation done.`);
        } else if (msg.type === "error") {
          throw new Error(msg.message || "Backend streaming error.");
        } else {
          runLogAppend(`[${nowTime()}] (info) ${line.slice(0, 120)}`);
        }
      }
    }

    if (!finalPayload || !finalPayload.analyses) {
      throw new Error("Streaming finished but no final payload was received.");
    }

    return finalPayload; // { type:"final", analyses:{...} }
  }

  // =========================
  // Rendering
  // =========================
  function renderSummary(analysis) {
    if (!analysis) return;

    if (elSummaryScore) elSummaryScore.textContent = `${analysis.overall_score} / 100`;

    const items = Array.isArray(analysis.items) ? analysis.items : [];
    const found = items.filter(x => x && x.present).length;
    if (elSummaryFound) elSummaryFound.textContent = `${found} of ${items.length}`;

    const cons = analysis.meta?.consistency;
    if (elSummaryConsistency) {
      elSummaryConsistency.textContent =
        (typeof cons === "number" && Number.isFinite(cons)) ? cons.toFixed(2) : "—";
    }

    const st = analysis.meta?.setup;
    if (elSummarySetup) {
      elSummarySetup.textContent = st
        ? `${st.clts} verbal CLTs · ${st.llm_runs} LLM runs · ${st.mode}`
        : "—";
    }

    const modeStr = analysis.meta?.setup?.mode || "";
    if (elRunInfo) elRunInfo.textContent = modeStr ? `Model: ${modeStr}` : "—";
  }

  function renderStrengthPips(strength, max = 3) {
    const p = [];
    for (let i = 1; i <= max; i++) p.push(`<span class="pip ${i <= strength ? "on" : "off"}"></span>`);
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
        <td><span class="badge">Open</span></td>
      `;

      tr.addEventListener("click", () => onSelect(it.clt_id));
      elTbody.appendChild(tr);
    });
  }

  function renderSpeechPanel(text, analysis, selectedCltId) {
    if (!elSpeechPanel) return;

    const sentences = splitSentences(text);
    const items = Array.isArray(analysis.items) ? analysis.items : [];

    const evidMap = new Map();
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
      div.innerHTML = `<small>Sentence ${idx + 1}</small>${html}`;

      if (selectedCltId && div.querySelector(`[data-clt="${selectedCltId}"]`)) {
        div.style.borderColor = "rgba(106,167,255,.35)";
        div.style.background = "rgba(106,167,255,.05)";
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

    if (!sentences.length) elSpeechPanel.innerHTML = `<div class="muted">No text found (Step 1).</div>`;
  }

  function renderDetails(item) {
    if (!elDetailPanel || !elDetailTitle || !elDetailMeta || !elDetailBody) return;

    if (!item) {
      elDetailPanel.dataset.open = "false";
      elDetailTitle.textContent = "Details";
      elDetailMeta.textContent = "Select a CLT.";
      elDetailBody.innerHTML = `<div class="muted">No selection yet.</div>`;
      return;
    }

    elDetailPanel.dataset.open = "true";
    elDetailTitle.textContent = item.label;
    elDetailMeta.textContent = `Strength: ${item.strength}/3 · Confidence: ${Number(item.confidence || 0).toFixed(2)}`;

    const ev = (item.evidence || []).map(e => e?.quote).filter(Boolean);
    const evHtml = ev.length
      ? `<ul>${ev.map(q => `<li><span class="badge">Evidence</span> <span class="muted">“${escapeHtml(q)}”</span></li>`).join("")}</ul>`
      : `<div class="muted">No evidence (no hit).</div>`;

    elDetailBody.innerHTML = `
      <details open class="card" style="padding:10px; margin-bottom:10px;">
        <summary style="cursor:pointer; font-weight:700;">Evidence</summary>
        <div style="margin-top:8px;">${evHtml}</div>
      </details>

      <details open class="card" style="padding:10px; margin-bottom:10px;">
        <summary style="cursor:pointer; font-weight:700;">Rationale</summary>
        <div class="muted" style="margin-top:8px;">${escapeHtml(item.rationale_short || "—")}</div>
      </details>

      
    `;
  }

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
      btnToggleLLM.textContent = (model === "gpt") ? "Select LLM: Gemini" : "Select LLM: GPT";
    }

    STATE.selected = null;
    updateSelectedRow(null);

    renderAll();
  }

  function showError(err) {
    const msg = (err && err.message) ? err.message : String(err);

    if (elSummaryScore) elSummaryScore.textContent = "—";
    if (elSummaryFound) elSummaryFound.textContent = "—";
    if (elSummaryConsistency) elSummaryConsistency.textContent = "—";
    if (elSummarySetup) elSummarySetup.textContent = "Backend error";
    if (elRunInfo) elRunInfo.textContent = "—";

    if (getHybridFlag()) runLogAppend(`[${nowTime()}] ERROR: ${msg}`);

    if (elTbody) elTbody.innerHTML = `<tr><td colspan="5" class="muted">Analysis failed: ${escapeHtml(msg)}</td></tr>`;
    if (elSpeechPanel) elSpeechPanel.innerHTML = `<div class="muted">Analysis failed: ${escapeHtml(msg)}</div>`;

    renderDetails(null);
  }

  async function runAnalysis() {
    const text = safeLocalStorageGet(TEXT_KEY) || "";
    STATE.text = text;

    const enableCompare = getCompareFlag();
    const useHybrid = getHybridFlag();

    // Show run box only in hybrid mode
    setRunBoxVisible(!!useHybrid);

    setLoading(true, useHybrid ? "Analyzing (hybrid)..." : "Analyzing...");

    try {
      if (useHybrid) {
        // Stream mode
        const final = await streamAnalysisFromBackend(text, enableCompare, useHybrid);
        STATE.analyses.gpt = final?.analyses?.gpt || null;
        STATE.analyses.gemini = final?.analyses?.gemini || null;
      } else {
        // Non-stream mode
        const data = await fetchAnalysisFromBackend(text, enableCompare, useHybrid);
        STATE.analyses.gpt = data?.analyses?.gpt || null;
        STATE.analyses.gemini = data?.analyses?.gemini || null;
      }

      if (btnToggleLLM) {
        if (!STATE.analyses.gemini) {
          btnToggleLLM.disabled = true;
          btnToggleLLM.textContent = "Select LLM: Gemini (not available)";
        } else {
          btnToggleLLM.disabled = false;
        }
      }

      setActiveModel("gpt");
      renderDetails(null);

      if (btnReanalyze) btnReanalyze.disabled = false;

      if (btnToggleLLM && !btnToggleLLM.dataset.bound) {
        btnToggleLLM.dataset.bound = "1";
        btnToggleLLM.addEventListener("click", () => {
          const next = (STATE.activeModel === "gpt") ? "gemini" : "gpt";
          if (STATE.analyses[next]) setActiveModel(next);
        });
      }
    } catch (err) {
      console.error(err);
      showError(err);
    } finally {
      setLoading(false);
    }
  }

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

  if (btnReanalyze) btnReanalyze.disabled = true;
  runAnalysis();
})();