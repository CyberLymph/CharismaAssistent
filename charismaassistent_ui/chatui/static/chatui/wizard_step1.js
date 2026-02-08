(function () {
  const STORAGE_KEY = "charismaassistant:speechText:v1";

  const elText = document.getElementById("speechText");
  const elPreview = document.getElementById("preview");
  const elCounter = document.getElementById("counter");
  const btnClear = document.getElementById("btnClear");
  const btnExample = document.getElementById("btnExample");
  const btnNext = document.getElementById("btnNext");

  function wordCount(text) {
    // Count words in a robust way (trim + split on whitespace)
    const t = (text || "").trim();
    if (!t) return 0;
    return t.split(/\s+/).length;
  }

  function updateUI() {
    const text = elText.value || "";
    const words = wordCount(text);
    const chars = text.length;

    elCounter.textContent = `${words} Wörter · ${chars} Zeichen`;
    elPreview.textContent = (text.trim() ? text.slice(0, 600) : "—");

    // Enable next only when meaningful text exists
    btnNext.disabled = words < 3; // small threshold to avoid empty submits

    // Persist for Step 2
    try {
      localStorage.setItem(STORAGE_KEY, text);
    } catch (e) {
      // ignore if storage blocked
    }
  }

  function loadFromStorage() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && !elText.value) {
        elText.value = saved;
      }
    } catch (e) {}
    updateUI();
  }

  btnClear.addEventListener("click", () => {
    elText.value = "";
    updateUI();
    elText.focus();
  });

  btnExample.addEventListener("click", () => {
    const sample =
`Liebe Kolleginnen und Kollegen,
wir stehen heute an einem Wendepunkt. Werden wir zuschauen – oder gestalten?
Ich verspreche Ihnen: Wir werden dieses Land nicht nur verwalten, wir werden es erneuern.

Wenn wir mutig sind, gewinnen wir Vertrauen. Wenn wir zögern, verlieren wir Zeit.
Nicht irgendwann. Nicht später. Jetzt.

Ich frage Sie: Wollen wir eine Zukunft, die uns passiert – oder eine Zukunft, die wir möglich machen?
Gemeinsam werden wir zeigen, dass Fortschritt nicht Angst macht, sondern Hoffnung gibt.`;

    elText.value = sample;
    updateUI();
    elText.focus();
  });

  elText.addEventListener("input", updateUI);

  btnNext.addEventListener("click", () => {
    // Later: navigate to Step 2 (analysis page)
    // For now, keep it simple: go to future route placeholder.
    // You can implement wizard_step2 URL next and update this link.
    window.location.href = "/wizard/step-2/";
  });

  loadFromStorage();
})();
