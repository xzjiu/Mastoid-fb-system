const state = { bootstrap: null, selectedId: null, stopAt: null };

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function fmtTime(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${(total - minutes * 60).toFixed(1).padStart(4, "0")}`;
}

async function api(path) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" } });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}

function feedbackItems() {
  return state.bootstrap?.coaching_plan?.items || [];
}

function shortAction(item) {
  return item.recommended_action || item.message || "Review the linked interval before your next attempt.";
}

function renderRunSelector() {
  const select = document.getElementById("runSelect");
  const runs = state.bootstrap.available_runs?.length
    ? state.bootstrap.available_runs : [{ run_id: state.bootstrap.run.run_id, url: "" }];
  select.replaceChildren(...runs.map(run => {
    const selected = run.run_id === state.bootstrap.run.run_id;
    const option = new Option(run.run_id, run.run_id, selected, selected);
    option.dataset.url = run.url || "";
    return option;
  }));
  select.disabled = runs.length < 2;
  select.addEventListener("change", () => {
    const url = select.selectedOptions[0]?.dataset.url;
    if (url) window.location.assign(url);
  });
}

function renderProcedureMap() {
  const duration = Number(state.bootstrap.run.duration_s || 1);
  document.getElementById("durationLabel").textContent = `${fmtTime(duration)} total`;
  document.getElementById("timelineEnd").textContent = fmtTime(duration);
  document.getElementById("phaseMap").innerHTML = state.bootstrap.phases.map(phase => {
    const left = 100 * Number(phase.t_start_s) / duration;
    const width = 100 * (Number(phase.t_end_s) - Number(phase.t_start_s)) / duration;
    const name = String(phase.phase_name || "").replaceAll("_", " ");
    return `<div class="phase-segment ${escapeHtml(phase.phase_name)}" style="left:${left}%;width:${width}%" title="${escapeHtml(name)} · ${fmtTime(phase.t_start_s)}–${fmtTime(phase.t_end_s)}">${escapeHtml(name)}</div>`;
  }).join("");
  document.getElementById("coachingMap").innerHTML = feedbackItems().map(item => {
    const left = 100 * Number(item.t_start_s) / duration;
    return `<button class="map-marker ${escapeHtml(item.priority)}" style="left:${left}%" data-feedback-id="${escapeHtml(item.feedback_id)}" title="${escapeHtml(item.title)} · ${fmtTime(item.t_start_s)}"></button>`;
  }).join("");
  document.querySelectorAll("[data-feedback-id]").forEach(button => button.addEventListener("click", () => selectFeedback(button.dataset.feedbackId, true)));
}

function renderFocusList() {
  const root = document.getElementById("focusList");
  const items = feedbackItems();
  if (!items.length) {
    root.innerHTML = `<div class="empty-state"><p>No coaching priority was generated for this session.</p></div>`;
    return;
  }
  root.innerHTML = items.map((item, index) => `
    <button class="focus-button ${escapeHtml(item.priority)} ${item.feedback_id === state.selectedId ? "active" : ""}" data-focus-id="${escapeHtml(item.feedback_id)}">
      <span class="meta"><span>${index === 0 ? "Start here" : `Focus ${index + 1}`}</span><span>${escapeHtml(String(item.phase || "session").replaceAll("_", " "))}</span></span>
      <h3>${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.message || shortAction(item))}</p>
    </button>`).join("");
  root.querySelectorAll("[data-focus-id]").forEach(button => button.addEventListener("click", () => selectFeedback(button.dataset.focusId, false)));
}

function clipsFor(item) {
  const clips = item.review_clips || item.evidence_segments || [];
  return clips.length ? clips : [{ t_start_s: item.t_start_s, t_end_s: item.t_end_s }];
}

function renderDetail(item) {
  const mechanism = item.why_it_matters || "Review the linked interval to connect this observation with your next practice step.";
  document.getElementById("feedbackDetail").innerHTML = `
    <div class="detail-head">
      <div><p class="eyebrow">${escapeHtml(String(item.category || "coaching").replaceAll("_", " "))}</p><h2>${escapeHtml(item.title)}</h2><p>${escapeHtml(item.message || "")}</p></div>
      <span class="priority ${escapeHtml(item.priority)}">${escapeHtml(item.priority)} priority</span>
    </div>
    <div class="action-callout"><small>Try this next</small><p>${escapeHtml(shortAction(item))}</p></div>
    <div class="story-grid">
      <div class="story-block"><h3>What happened</h3><p>${escapeHtml(item.observed_issue || item.evidence_summary || "A time-linked coaching moment was identified.")}</p></div>
      <div class="story-block"><h3>Why it matters</h3><p>${escapeHtml(mechanism)}</p></div>
    </div>
    <button class="watch-button" id="watchSelected">Watch the linked moment</button>`;
  document.getElementById("watchSelected").addEventListener("click", () => playClip(clipsFor(item)[0], item.title, 0));
  renderClips(item);
}

function renderClips(item) {
  const clips = clipsFor(item);
  document.getElementById("clipList").innerHTML = clips.map((clip, index) => `
    <button data-clip-index="${index}">${clips.length > 1 ? `Moment ${index + 1} · ` : ""}${fmtTime(clip.t_start_s)}–${fmtTime(clip.t_end_s)}</button>`).join("");
  document.querySelectorAll("[data-clip-index]").forEach(button => button.addEventListener("click", () => {
    const index = Number(button.dataset.clipIndex);
    playClip(clips[index], item.title, index);
  }));
}

function selectFeedback(feedbackId, play) {
  const item = feedbackItems().find(row => row.feedback_id === feedbackId);
  if (!item) return;
  state.selectedId = feedbackId;
  renderFocusList();
  renderDetail(item);
  if (play) playClip(clipsFor(item)[0], item.title, 0);
}

function playClip(clip, title, index) {
  const video = document.getElementById("sessionVideo");
  if (!state.bootstrap.media?.available) return;
  const start = Math.max(0, Number(clip.t_start_s) - 5);
  const stop = Math.min(Number(state.bootstrap.run.duration_s), Number(clip.t_end_s) + 5);
  document.getElementById("videoHeading").textContent = title;
  document.getElementById("videoWindow").textContent = `${fmtTime(start)}–${fmtTime(stop)}`;
  document.querySelectorAll("[data-clip-index]").forEach(button => button.classList.toggle("active", Number(button.dataset.clipIndex) === index));
  const seek = () => {
    video.currentTime = start;
    state.stopAt = stop;
    video.play().catch(() => {});
    document.getElementById("videoCard").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (video.readyState >= 1) seek(); else video.addEventListener("loadedmetadata", seek, { once: true });
}

async function init() {
  try {
    state.bootstrap = await api("/api/bootstrap");
    renderRunSelector();
    renderProcedureMap();
    const video = document.getElementById("sessionVideo");
    if (state.bootstrap.media?.available) {
      video.src = state.bootstrap.media.url;
      video.addEventListener("timeupdate", () => {
        if (state.stopAt != null && video.currentTime >= state.stopAt) {
          video.pause(); state.stopAt = null;
        }
      });
    } else {
      video.classList.add("hidden");
      document.getElementById("videoUnavailable").classList.remove("hidden");
    }
    const first = feedbackItems()[0];
    if (first) selectFeedback(first.feedback_id, false); else renderFocusList();
  } catch (error) {
    document.querySelector("main").innerHTML = `<section class="feedback-detail empty-state"><h2>Could not load this session</h2><p>${escapeHtml(error.message)}</p></section>`;
  }
}

init();
