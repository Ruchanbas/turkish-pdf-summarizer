// ozetle arayüzü — derleme adımı yok, bağımlılık yok.
//
// Akış: POST /api/jobs → (gerekirse) dosyayı upload.url'e POST et →
// GET /api/jobs/{id} ile yokla. API yolu göreli: local'de FastAPI, AWS'de
// CloudFront aynı alan adında /api/* isteğini API Gateway'e yönlendirir.

"use strict";

const API = "api";
const POLL_MIN_MS = 1000;
const POLL_MAX_MS = 3000;
const GIVE_UP_MS = 12 * 60 * 1000;
const COLD_START_HINT_MS = 15000;

const $ = (id) => document.getElementById(id);
const els = {
  intake: $("intake"),
  dropzone: $("dropzone"),
  input: $("file-input"),
  intakeError: $("intake-error"),
  limits: $("limits"),
  progress: $("progress"),
  progressTitle: $("progress-title"),
  meter: document.querySelector(".meter"),
  status: $("progress-status"),
  coldStart: $("cold-start"),
  cancel: $("cancel"),
  result: $("result"),
  resultTitle: $("result-title"),
  facts: $("result-facts"),
  warnings: $("warnings"),
  summary: $("summary-text"),
  copy: $("copy"),
  download: $("download"),
  again: $("again"),
  copyStatus: $("copy-status"),
  keywordsBlock: $("keywords-block"),
  keywords: $("keywords"),
  sectionList: $("section-list"),
  modelName: $("model-name"),
};

const state = {
  maxBytes: 20 * 1024 * 1024,
  job: null,
  pollTimer: null,
  tickTimer: null,
  run: 0, // her yeni işte artar; eski yoklamalar kendini durdurur
};

// ------------------------------------------------------------ yardımcılar

const trLower = (s) => s.toLocaleLowerCase("tr");

function formatBytes(n) {
  return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(n / 1024)} KB`;
}

function formatClock(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

async function sha256Hex(file) {
  if (!window.crypto || !crypto.subtle) return null; // güvenli olmayan bağlam
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}/${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Sunucuya ulaşılamadı. İnternet bağlantınızı kontrol edip tekrar deneyin.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Sunucu hata döndürdü (${response.status}).`);
  }
  return body;
}

function show(section) {
  for (const el of [els.intake, els.progress, els.result]) el.hidden = el !== section;
}

function stopTimers() {
  clearTimeout(state.pollTimer);
  clearInterval(state.tickTimer);
}

// ------------------------------------------------------------ yükleme alanı

function showIntakeError(message) {
  show(els.intake);
  els.intakeError.textContent = message;
  els.intakeError.hidden = false;
}

function validate(file) {
  const isPdf = file.type === "application/pdf" || trLower(file.name).endsWith(".pdf");
  if (!isPdf) return "Bu dosya bir PDF değil. Yalnızca PDF dosyaları özetlenebilir.";
  if (file.size === 0) return "Dosya boş görünüyor.";
  if (file.size > state.maxBytes) {
    return `Dosya ${formatBytes(file.size)}; en fazla ${formatBytes(state.maxBytes)} yüklenebilir.`;
  }
  return null;
}

els.input.addEventListener("change", () => {
  const [file] = els.input.files;
  if (file) start(file);
  els.input.value = "";
});

for (const type of ["dragenter", "dragover"]) {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("is-over");
  });
}
for (const type of ["dragleave", "drop"]) {
  els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("is-over"));
}
els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  const [file] = event.dataTransfer.files;
  if (file) start(file);
});
// Sayfanın başka yerine bırakılan dosya tarayıcıda açılmasın.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ------------------------------------------------------------ ilerleme

const STAGES = ["upload", "queued", "processing", "done"];

function setStage(stage, message) {
  const current = STAGES.indexOf(stage);
  for (const li of els.progress.querySelectorAll(".stages li")) {
    const i = STAGES.indexOf(li.dataset.stage);
    li.classList.toggle("is-done", i < current);
    li.classList.toggle("is-current", i === current);
    if (i === current) li.setAttribute("aria-current", "step");
    else li.removeAttribute("aria-current");
  }
  if (message) els.status.textContent = message;
}

function setMeter(fraction) {
  const bar = els.meter.firstElementChild;
  els.meter.classList.toggle("is-indeterminate", fraction === null);
  bar.style.width = fraction === null ? "" : `${Math.round(fraction * 100)}%`;
  els.meter.setAttribute("aria-valuenow", fraction === null ? "" : String(Math.round(fraction * 100)));
}

function upload(target, file, onProgress) {
  // fetch yükleme ilerlemesi vermez; XHR verir.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(target.fields || {})) form.append(key, value);
    form.append("file", file); // S3 için dosya en sonda olmalı
    const xhr = new XMLHttpRequest();
    xhr.open("POST", target.url);
    xhr.upload.addEventListener("progress", (e) => e.lengthComputable && onProgress(e.loaded / e.total));
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve();
      let message = "Dosya yüklenemedi.";
      try {
        message = JSON.parse(xhr.responseText).error || message;
      } catch {
        if (xhr.responseText.includes("EntityTooLarge")) message = "Dosya boyut sınırını aşıyor.";
        else if (xhr.responseText.includes("expired")) message = "Yükleme süresi doldu. Tekrar deneyin.";
      }
      reject(new Error(message));
    });
    xhr.addEventListener("error", () => reject(new Error("Yükleme sırasında bağlantı koptu.")));
    xhr.send(form);
  });
}

async function start(file) {
  const problem = validate(file);
  if (problem) return showIntakeError(problem);

  stopTimers();
  const run = ++state.run;
  els.intakeError.hidden = true;
  els.coldStart.hidden = true;
  els.progressTitle.textContent = file.name;
  show(els.progress);
  setStage("upload", "Dosya hazırlanıyor…");
  setMeter(0);

  try {
    const sha256 = await sha256Hex(file);
    const created = await api("jobs", {
      method: "POST",
      body: JSON.stringify({ filename: file.name, size: file.size, sha256 }),
    });
    if (run !== state.run) return;
    history.replaceState(null, "", `#is=${created.job.id}`);

    if (created.upload) {
      setStage("upload", `Yükleniyor (${formatBytes(file.size)})`);
      await upload(created.upload, file, (f) => run === state.run && setMeter(f));
      if (run !== state.run) return;
    }
    follow(created.job.id, run);
  } catch (error) {
    if (run === state.run) showIntakeError(error.message);
  }
}

function follow(jobId, run, startedAt = Date.now()) {
  setMeter(null);
  setStage("queued", "Sırada bekliyor");
  clearInterval(state.tickTimer);
  state.tickTimer = setInterval(() => {
    const elapsed = Date.now() - startedAt;
    if (state.job?.status === "processing") {
      els.status.textContent = `Özetleniyor, geçen süre ${formatClock(elapsed)}`;
      els.coldStart.hidden = elapsed < COLD_START_HINT_MS;
    }
  }, 1000);

  let delay = POLL_MIN_MS;
  const poll = async () => {
    if (run !== state.run) return;
    try {
      const job = await api(`jobs/${jobId}`);
      if (run !== state.run) return;
      state.job = job;
      els.progressTitle.textContent = job.filename;
      if (job.status === "done") return finish(job);
      if (job.status === "failed") {
        stopTimers();
        history.replaceState(null, "", location.pathname);
        return showIntakeError(job.error || "Belge işlenemedi.");
      }
      if (job.status === "processing") {
        setStage("processing", `Özetleniyor, geçen süre ${formatClock(Date.now() - startedAt)}`);
      } else {
        setStage("queued", "Sırada bekliyor");
      }
    } catch (error) {
      if (/bulunamadı/i.test(error.message)) {
        stopTimers();
        history.replaceState(null, "", location.pathname);
        return showIntakeError(error.message);
      }
      // Geçici ağ hatası: yoklamaya devam.
    }
    if (Date.now() - startedAt > GIVE_UP_MS) {
      stopTimers();
      return showIntakeError("İşlem beklenenden uzun sürdü. Sayfayı yenileyip daha sonra tekrar bakın.");
    }
    delay = Math.min(POLL_MAX_MS, delay * 1.3);
    state.pollTimer = setTimeout(poll, delay);
  };
  poll();
}

els.cancel.addEventListener("click", reset);

// ------------------------------------------------------------ sonuç

/** Metni, anahtar kelime geçişleri <mark> içinde olacak şekilde DOM'a yazar.
 *  innerHTML kullanılmaz: model çıktısı asla HTML olarak yorumlanmaz. */
function renderHighlighted(container, text, phrases) {
  container.replaceChildren();
  const lowered = trLower(text); // tr yerel ayarı uzunluğu korur (İ→i, I→ı)
  const needles = phrases
    .map((p) => trLower(p.trim()))
    .filter((p) => p.length > 2)
    .sort((a, b) => b.length - a.length);
  let pos = 0;
  let markIndex = 0;
  while (pos < text.length) {
    let hit = null;
    for (const needle of needles) {
      const at = lowered.indexOf(needle, pos);
      if (at !== -1 && (hit === null || at < hit.at)) hit = { at, len: needle.length };
    }
    if (hit === null || lowered.length !== text.length) {
      container.append(text.slice(pos));
      break;
    }
    container.append(text.slice(pos, hit.at));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(hit.at, hit.at + hit.len);
    mark.style.setProperty("--i", String(markIndex++));
    container.append(mark);
    pos = hit.at + hit.len;
  }
  return markIndex;
}

function finish(job) {
  stopTimers();
  setStage("done", "Hazır");
  const r = job.result;
  const stats = r.stats;
  els.resultTitle.textContent = job.filename;

  const facts = [
    `${stats.pages_read} sayfa okundu`,
    `${stats.chunks} bölüm`,
    stats.rounds > 1 ? `${stats.rounds} turda birleştirildi` : null,
    stats.seconds?.summarize != null ? `${Math.round(stats.seconds.summarize)} saniyede özetlendi` : null,
    job.cached ? "daha önce özetlenmiş, önbellekten geldi" : null,
  ].filter(Boolean);
  els.facts.textContent = facts.join(", ") + ".";

  els.warnings.replaceChildren();
  els.warnings.hidden = r.warnings.length === 0;
  if (r.warnings.length) {
    const list = document.createElement("ul");
    for (const w of r.warnings) {
      const li = document.createElement("li");
      li.textContent = w;
      list.append(li);
    }
    els.warnings.append(list);
  }

  const phrases = r.keywords.map((k) => k.phrase);
  renderHighlighted(els.summary, r.summary, phrases);

  els.keywords.replaceChildren();
  els.keywordsBlock.hidden = r.keywords.length === 0;
  r.keywords.forEach((k, index) => {
    const li = document.createElement("li");
    const mark = document.createElement("mark");
    mark.textContent = k.phrase;
    mark.title = `Benzerlik ${k.score.toFixed(2)}`;
    mark.style.setProperty("--i", String(index + 6));
    li.append(mark);
    els.keywords.append(li);
  });

  els.sectionList.replaceChildren();
  for (const s of r.sections) {
    const li = document.createElement("li");
    const source = document.createElement("p");
    source.className = "section-source";
    source.textContent = s.source;
    const note = document.createElement("p");
    note.className = "section-note";
    note.textContent = s.summary;
    li.append(source, note);
    els.sectionList.append(li);
  }

  show(els.result);
  els.result.classList.add("is-revealing");
  els.resultTitle.setAttribute("tabindex", "-1");
  els.resultTitle.focus({ preventScroll: false });
  window.scrollTo({ top: 0 });
}

function markdown(job) {
  const r = job.result;
  const lines = [`# ${job.filename}`, "", r.summary, ""];
  if (r.keywords.length) {
    lines.push(`**Anahtar kelimeler:** ${r.keywords.map((k) => k.phrase).join(", ")}`, "");
  }
  if (r.sections.length > 1) {
    lines.push("## Bölüm özetleri", "");
    r.sections.forEach((s, i) => lines.push(`${i + 1}. ${s.summary}`));
    lines.push("");
  }
  lines.push(`_Model: ${r.stats.model}. Otomatik üretilmiş özettir; alıntılamadan önce kaynağı kontrol edin._`, "");
  return lines.join("\n");
}

els.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.job.result.summary);
    els.copyStatus.textContent = "Özet panoya kopyalandı.";
  } catch {
    els.copyStatus.textContent = "Kopyalanamadı. Metni seçip elle kopyalayın.";
  }
});

els.download.addEventListener("click", () => {
  const blob = new Blob([markdown(state.job)], { type: "text/markdown;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = state.job.filename.replace(/\.pdf$/i, "") + "-ozet.md";
  link.click();
  URL.revokeObjectURL(link.href);
  els.copyStatus.textContent = "Markdown dosyası indirildi.";
});

function reset() {
  stopTimers();
  state.run++;
  state.job = null;
  els.result.classList.remove("is-revealing");
  els.copyStatus.textContent = "";
  history.replaceState(null, "", location.pathname);
  els.intakeError.hidden = true;
  show(els.intake);
  els.dropzone.focus?.();
}

els.again.addEventListener("click", reset);

// ------------------------------------------------------------ açılış

async function init() {
  try {
    const health = await api("health");
    state.maxBytes = health.limits.max_upload_mb * 1024 * 1024;
    els.limits.textContent = `En fazla ${health.limits.max_upload_mb} MB, ilk ${health.limits.max_pages} sayfa okunur.`;
    els.modelName.textContent = `Model: ${health.model}`;
  } catch {
    showIntakeError("Servis şu an yanıt vermiyor. Biraz sonra tekrar deneyin.");
  }

  // Sayfa yenilenirse veya bağlantı paylaşılırsa iş kaldığı yerden izlenir.
  const match = location.hash.match(/^#is=([0-9a-f]{32})$/);
  if (match) {
    els.progressTitle.textContent = "Belge";
    show(els.progress);
    follow(match[1], ++state.run);
  }
}

init();
