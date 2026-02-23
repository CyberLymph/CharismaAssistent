// wizard_step1.js (EN + compare + hybrid)
(function () {
  const TEXT_KEY = "charismaassistant:speechText:v1";
  const COMPARE_KEY = "charismaassistant:enableCompare:v1";
  const HYBRID_KEY = "charismaassistant:enableHybrid:v1";

  const elText = document.getElementById("speechText");
  const elPreview = document.getElementById("preview");
  const elCounter = document.getElementById("counter");
  const btnClear = document.getElementById("btnClear");
  const btnExample = document.getElementById("btnExample");
  const btnNext = document.getElementById("btnNext");

  const elCompare = document.getElementById("enableCompare");
  const elHybrid = document.getElementById("enableHybrid");

  function safeGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }
  function safeSet(key, value) {
    try { localStorage.setItem(key, value); } catch {}
  }

  function wordCount(text) {
    const t = (text || "").trim();
    if (!t) return 0;
    return t.split(/\s+/).length;
  }

  function updateUI() {
    const text = elText.value || "";
    const words = wordCount(text);
    const chars = text.length;

    if (elCounter) elCounter.textContent = `${words} words · ${chars} chars`;
    if (elPreview) elPreview.textContent = (text.trim() ? text.slice(0, 600) : "—");

    if (btnNext) btnNext.disabled = words < 3;

    safeSet(TEXT_KEY, text);
  }

  function loadFlags() {
    if (elCompare) elCompare.checked = safeGet(COMPARE_KEY) === "1";
    if (elHybrid) elHybrid.checked = safeGet(HYBRID_KEY) === "1";
  }

  function bindFlags() {
    if (elCompare) {
      elCompare.addEventListener("change", () => {
        safeSet(COMPARE_KEY, elCompare.checked ? "1" : "0");
      });
    }
    if (elHybrid) {
      elHybrid.addEventListener("change", () => {
        safeSet(HYBRID_KEY, elHybrid.checked ? "1" : "0");
      });
    }
  }

  function loadFromStorage() {
    const saved = safeGet(TEXT_KEY);
    if (saved && elText && !elText.value) elText.value = saved;
    updateUI();
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      elText.value = "";
      updateUI();
      elText.focus();
    });
  }

  if (btnExample) {
    btnExample.addEventListener("click", () => {
      const sample =
`My friends, today we stand at the edge of history. The question before us is simple: will we accept the limits others have placed upon us, or will we rise beyond them?
For years, families have worked hard, played by the rules, and still been denied the dignity they deserve.
We are not divided strangers. We are one people. One voice. One destiny. And together, together, together — we will not be ignored.
Some say change is impossible. But I say injustice is impossible to ignore.
This moment is a spark in the darkness. It is the sunrise after a long night.
We will build schools that inspire. We will create jobs that sustain. We will restore dignity where it has been denied.
And we will succeed — because we believe in our cause, because we believe in each other, and because we know that history bends toward those who refuse to surrender.`;

      elText.value = sample;
      updateUI();
      elText.focus();
    });
  }

  if (elText) elText.addEventListener("input", updateUI);

  if (btnNext) {
    btnNext.addEventListener("click", () => {
      window.location.href = "/wizard/step-2/";
    });
  }

  loadFlags();
  bindFlags();
  loadFromStorage();
})();