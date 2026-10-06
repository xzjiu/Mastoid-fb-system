const state = {
  participantId: null,
  start: null,
  experience: null,
  selectedId: null,
  stopAt: null,
  currentClipIndex: null,
  videoBound: false,
  expertMoments: [],
  momentTime: 0,
  mediaRecorder: null,
  audioChunks: [],
  audioBlob: null,
  audioDuration: 0,
  audioStartedAt: null,
  audioTimer: null,
  audioPreviewUrl: null,
  referenceStopAt: null,
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function fmtTime(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${(total - minutes * 60).toFixed(1).padStart(4, "0")}`;
}

function wait(milliseconds) {
  return new Promise(resolve => window.setTimeout(resolve, milliseconds));
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}

async function firstCaseUrlForAccess(participantId) {
  const response = await fetch("/api/health");
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Unable to verify the study entry point");
  const firstCase = (value.available_runs || [])[0];
  if (!firstCase?.url || firstCase.run_id === value.run_id) return null;
  const firstUrl = new URL(firstCase.url, window.location.href);
  firstUrl.searchParams.set("participant_id", participantId);
  return firstUrl.toString();
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function logEvent(eventType, details = {}, feedbackId = state.selectedId) {
  if (!state.participantId) return;
  api("/api/expert-study/event", {
    participant_id: state.participantId,
    stimulus_id: feedbackId || null,
    event_type: eventType,
    details,
  }).catch(() => {});
}

function setHeaderStatus(text) {
  document.getElementById("workflowStatus").textContent = text;
}

function caseHeader() {
  const value = state.start.case;
  document.getElementById("caseLabel").textContent = `${value.alias} of ${value.total}`;
}

async function enterStudy(participantId) {
  const value = await api("/api/expert-study/start", { participant_id: participantId });
  state.participantId = value.participant_id;
  state.start = value;
  state.expertMoments = value.expert_proposed_moments || [];
  document.getElementById("interfaceVersion").textContent = String(
    value.interface_version || "pilot-ui-v1.4.1"
  ).replaceAll("-", " ");
  const currentUrl = new URL(window.location.href);
  currentUrl.searchParams.set("participant_id", state.participantId);
  window.history.replaceState({}, "", currentUrl);
  document.getElementById("accessPanel").classList.add("hidden");
  document.getElementById("headerStatus").classList.remove("hidden");
  caseHeader();
  if (value.case_completed) {
    showCompletion();
    return;
  }
  setHeaderStatus("Session ready");
  document.getElementById("workflowDisclosure").textContent = value.workflow.disclosure;
  document.getElementById("readyCaseTitle").textContent = `${value.case.alias} is ready for post-session analysis`;
  document.getElementById("readyPanel").classList.remove("hidden");
}

function renderProcessingStages(stages) {
  document.getElementById("processingStages").innerHTML = stages.map((label, index) => `
    <div class="processing-stage" data-stage-index="${index}"><span>${index + 1}</span><strong>${escapeHtml(label)}</strong></div>
  `).join("");
}

async function animateProcessing(stages) {
  for (let index = 0; index < stages.length; index += 1) {
    const rows = document.querySelectorAll("[data-stage-index]");
    rows.forEach((row, rowIndex) => {
      row.classList.toggle("active", rowIndex === index);
      row.classList.toggle("done", rowIndex < index);
      row.querySelector("span").textContent = rowIndex < index ? "✓" : String(rowIndex + 1);
    });
    await wait(650);
  }
  document.querySelectorAll("[data-stage-index]").forEach(row => {
    row.classList.remove("active");
    row.classList.add("done");
    row.querySelector("span").textContent = "✓";
  });
  await wait(350);
}

async function startAnalysis() {
  const button = document.getElementById("startAnalysisButton");
  button.disabled = true;
  document.getElementById("readyPanel").classList.add("hidden");
  document.getElementById("processingPanel").classList.remove("hidden");
  setHeaderStatus("Analyzing recorded session");
  const stages = state.start.workflow.stages;
  renderProcessingStages(stages);
  try {
    const [experience] = await Promise.all([
      api("/api/expert-study/analyze", { participant_id: state.participantId }),
      animateProcessing(stages),
    ]);
    state.experience = experience;
    state.expertMoments = experience.expert_proposed_moments || [];
    document.getElementById("processingPanel").classList.add("hidden");
    document.getElementById("experiencePanel").classList.remove("hidden");
    setHeaderStatus("Feedback ready");
    renderExperience();
  } catch (error) {
    document.getElementById("processingPanel").classList.add("hidden");
    document.getElementById("readyPanel").classList.remove("hidden");
    button.disabled = false;
    setHeaderStatus("Session ready");
    showToast(error.message);
  }
}

function feedbackItems() {
  return state.experience?.feedback || [];
}

function renderProcedureMap() {
  const duration = Number(state.experience.session.duration_s || 1);
  document.getElementById("durationLabel").textContent = `${fmtTime(duration)} total`;
  document.getElementById("timelineEnd").textContent = fmtTime(duration);
  document.getElementById("phaseMap").innerHTML = state.experience.phases.map(phase => {
    const left = 100 * Number(phase.t_start_s) / duration;
    const width = 100 * (Number(phase.t_end_s) - Number(phase.t_start_s)) / duration;
    const name = String(phase.phase_name || "").replaceAll("_", " ");
    return `<div class="phase-segment ${escapeHtml(phase.phase_name)}" style="left:${left}%;width:${width}%" title="${escapeHtml(name)} · ${fmtTime(phase.t_start_s)}–${fmtTime(phase.t_end_s)}">${escapeHtml(name)}</div>`;
  }).join("");
  const automaticMarkers = feedbackItems().map(item => {
    const left = 100 * Number(item.t_start_s) / duration;
    return `<button class="map-marker ${escapeHtml(item.priority)}" style="left:${left}%" data-map-id="${escapeHtml(item.feedback_id)}" aria-label="Open coaching moment at ${fmtTime(item.t_start_s)}"></button>`;
  }).join("");
  const expertMarkers = state.expertMoments.map(item => {
    const left = 100 * Number(item.t_video_s) / duration;
    return `<button class="map-marker expert" style="left:${left}%" data-expert-time="${Number(item.t_video_s)}" aria-label="Open expert-added coaching moment at ${fmtTime(item.t_video_s)}"></button>`;
  }).join("");
  document.getElementById("coachingMap").innerHTML = automaticMarkers + expertMarkers;
  document.querySelectorAll("[data-map-id]").forEach(button => button.addEventListener("click", () => {
    selectFeedback(button.dataset.mapId, true, "procedure_map");
  }));
  document.querySelectorAll("[data-expert-time]").forEach(button => button.addEventListener("click", () => {
    const video = document.getElementById("sessionVideo");
    video.pause();
    video.currentTime = Number(button.dataset.expertTime);
    document.getElementById("videoCard").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

function renderFocusList() {
  const root = document.getElementById("focusList");
  const items = feedbackItems();
  if (!items.length) {
    root.innerHTML = `<div class="empty-state"><p>No automatic coaching priority was generated for this session.</p></div>`;
    return;
  }
  root.innerHTML = items.map((item, index) => `
    <button class="focus-button ${escapeHtml(item.priority)} ${item.feedback_id === state.selectedId ? "active" : ""}" data-focus-id="${escapeHtml(item.feedback_id)}">
      <span class="meta"><span>${index === 0 ? "Start here" : `Focus ${index + 1}`}</span><span>${escapeHtml(String(item.phase || "session").replaceAll("_", " "))}</span></span>
      <h3>${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.message || item.recommended_action || "Review the linked evidence.")}</p>
    </button>`).join("");
  root.querySelectorAll("[data-focus-id]").forEach(button => button.addEventListener("click", () => {
    selectFeedback(button.dataset.focusId, false, "feedback_list");
  }));
}

function renderDetail(item) {
  const method = item.method_details || {};
  document.getElementById("feedbackDetail").innerHTML = `
    <div class="detail-head">
      <div><p class="eyebrow">${escapeHtml(String(item.category || "coaching").replaceAll("_", " "))}</p><h2>${escapeHtml(item.title)}</h2><p>${escapeHtml(item.message || "")}</p></div>
      <span class="priority ${escapeHtml(item.priority)}">${escapeHtml(item.priority)} priority</span>
    </div>
    <div class="action-callout"><small>Try this next</small><p>${escapeHtml(item.recommended_action || item.message || "Review the linked interval before the next attempt.")}</p></div>
    <div class="story-grid">
      <div class="story-block"><h3>What happened</h3><p>${escapeHtml(item.observed_issue || item.evidence_summary || "A time-linked coaching moment was identified.")}</p></div>
      <div class="story-block"><h3>Why it matters</h3><p>${escapeHtml(item.why_it_matters || "Review the linked evidence in the context of the next practice attempt.")}</p></div>
    </div>
    <details class="item-method">
      <summary>Why this feedback is shown</summary>
      <div class="method-grid">
        <div><strong>What the system reviewed</strong><p>${escapeHtml(method.data_used || "Time-aligned simulator evidence.")}</p></div>
        <div><strong>What it looked for</strong><p>${escapeHtml(method.trigger || item.evidence_summary || "A predefined evidence rule was met.")}</p></div>
        <div><strong>Why these examples</strong><p>${escapeHtml(method.selection || "Representative review locations are shown first.")}</p></div>
        <div><strong>Interpretation boundary</strong><p>${escapeHtml(method.limitation || item.claim_boundary || "Review support only.")}</p></div>
      </div>
    </details>
    <button class="watch-button" id="watchSelected">Watch the linked moment</button>`;
  document.getElementById("watchSelected").addEventListener("click", () => playClip(0, true));
  renderClips(item);
}

function anatomyLabel(value) {
  return {
    FacialNerve: "Facial nerve",
    SigSinus: "Sigmoid sinus",
    BonyLabyrinth: "Bony labyrinth",
  }[String(value || "")] || String(value || "Other review moments").replaceAll("_", " ");
}

function clipRowHtml(entry, label) {
  const clip = entry.clip;
  const signalLabel = clip.exact_signal_frame
    ? `Key frame · ${fmtTime(clip.signal_time_s)}`
    : `Review interval ${fmtTime(clip.t_start_s)}–${fmtTime(clip.t_end_s)}`;
  return `<div class="clip-row">
    <div class="clip-copy"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(signalLabel)}</span></div>
    <div class="clip-actions">
      ${clip.exact_signal_frame ? `<button data-signal-index="${entry.index}">Go to key frame</button>` : ""}
      <button data-clip-index="${entry.index}">Play context</button>
    </div>
  </div>`;
}

function renderClips(item) {
  const clips = item.clips || [];
  const root = document.getElementById("clipList");
  const indexed = clips.map((clip, index) => ({ clip, index }));
  if (item.category === "safety") {
    const grouped = new Map();
    indexed.forEach(entry => {
      const key = entry.clip.anatomy || "Other";
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(entry);
    });
    const groups = [...grouped.entries()].map(([anatomy, entries]) => ({ anatomy, entries }));
    root.innerHTML = groups.map(group => {
      const first = group.entries[0];
      const rest = group.entries.slice(1);
      return `<section class="clip-group">
        <div class="clip-group-head"><strong>${escapeHtml(anatomyLabel(group.anatomy))}</strong><span>${group.entries.length} ${group.entries.length === 1 ? "moment" : "moments"}</span></div>
        ${clipRowHtml(first, "Representative moment")}
        ${rest.length ? `<details class="additional-moments"><summary>Show ${rest.length} similar ${rest.length === 1 ? "moment" : "moments"}</summary>${rest.map((entry, index) => clipRowHtml(entry, `Additional moment ${index + 1}`)).join("")}</details>` : ""}
      </section>`;
    }).join("");
  } else if (item.category === "exposure") {
    const grouped = new Map();
    indexed.forEach(entry => {
      const key = entry.clip.phase || "Other";
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(entry);
    });
    root.innerHTML = [...grouped.entries()].map(([phase, entries]) => {
      const first = entries[0];
      const rest = entries.slice(1);
      const phaseLabel = String(phase || "Procedure").replaceAll("_", " ");
      return `<section class="clip-group">
        <div class="clip-group-head"><strong>${escapeHtml(phaseLabel)}</strong><span>${entries.length} ${entries.length === 1 ? "example" : "examples"}</span></div>
        ${clipRowHtml(first, "Start with this example")}
        ${rest.length ? `<details class="additional-moments"><summary>Show ${rest.length} more from this phase</summary>${rest.map((entry, index) => clipRowHtml(entry, `Additional example ${index + 1}`)).join("")}</details>` : ""}
      </section>`;
    }).join("");
  } else {
    root.innerHTML = `<section class="clip-group">${indexed.map((entry, index) => clipRowHtml(entry, clips.length > 1 ? `Review moment ${index + 1}` : "Review moment")).join("")}</section>`;
  }
  document.querySelectorAll("[data-clip-index]").forEach(button => button.addEventListener("click", () => {
    playClip(Number(button.dataset.clipIndex), false);
  }));
  document.querySelectorAll("[data-signal-index]").forEach(button => button.addEventListener("click", () => {
    jumpToSignal(Number(button.dataset.signalIndex), true);
  }));
}

function selectFeedback(feedbackId, play, source) {
  const item = feedbackItems().find(row => row.feedback_id === feedbackId);
  if (!item) return;
  state.selectedId = feedbackId;
  state.currentClipIndex = null;
  renderFocusList();
  renderDetail(item);
  logEvent("feedback_item_opened", { source, title: item.title }, feedbackId);
  if (play) playClip(0, true);
}

function playClip(index, scrollToVideo) {
  const item = feedbackItems().find(row => row.feedback_id === state.selectedId);
  const clip = item?.clips?.[index];
  const video = document.getElementById("sessionVideo");
  if (!item || !clip || !state.experience.media?.available) return;
  const mediaEnd = Number(state.experience.media.duration_s || clip.playback_end_s);
  const start = Math.max(0, Number(clip.playback_start_s));
  const stop = Math.min(mediaEnd, Number(clip.playback_end_s));
  state.currentClipIndex = index;
  state.stopAt = stop;
  document.getElementById("videoHeading").textContent = item.title;
  document.getElementById("videoWindow").textContent = `${fmtTime(start)}–${fmtTime(stop)}`;
  document.querySelectorAll("[data-clip-index], [data-signal-index]").forEach(button => {
    const buttonIndex = button.dataset.clipIndex ?? button.dataset.signalIndex;
    button.classList.toggle("active", Number(buttonIndex) === index);
  });
  const seek = () => {
    video.currentTime = start;
    video.play().catch(() => {});
    if (scrollToVideo) document.getElementById("videoCard").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, { once: true });
  logEvent("feedback_clip_played", { clip_index: index, playback_start_s: start, playback_end_s: stop });
}

function jumpToSignal(index, scrollToVideo) {
  const item = feedbackItems().find(row => row.feedback_id === state.selectedId);
  const clip = item?.clips?.[index];
  const video = document.getElementById("sessionVideo");
  if (!item || !clip || !clip.exact_signal_frame || !state.experience.media?.available) return;
  const signalTime = Number(clip.signal_time_s);
  state.currentClipIndex = index;
  state.stopAt = null;
  document.getElementById("videoHeading").textContent = `${anatomyLabel(clip.anatomy)} review frame`;
  document.getElementById("videoWindow").textContent = `Key frame · ${fmtTime(signalTime)}`;
  document.querySelectorAll("[data-clip-index], [data-signal-index]").forEach(button => {
    const buttonIndex = button.dataset.clipIndex ?? button.dataset.signalIndex;
    button.classList.toggle("active", Number(buttonIndex) === index);
  });
  const seek = () => {
    video.pause();
    video.currentTime = signalTime;
    if (scrollToVideo) document.getElementById("videoCard").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, { once: true });
  logEvent("feedback_signal_frame_opened", {
    clip_index: index,
    signal_time_s: signalTime,
    signal_frame_idx: clip.signal_frame_idx,
    anatomy: clip.anatomy,
  });
}

function bindVideo() {
  if (state.videoBound) return;
  state.videoBound = true;
  const video = document.getElementById("sessionVideo");
  video.addEventListener("play", () => logEvent("video_play", {
    current_time_s: Number(video.currentTime.toFixed(3)), clip_index: state.currentClipIndex,
  }));
  video.addEventListener("pause", () => logEvent("video_pause", {
    current_time_s: Number(video.currentTime.toFixed(3)), clip_index: state.currentClipIndex,
  }));
  video.addEventListener("seeked", () => logEvent("video_seek", {
    current_time_s: Number(video.currentTime.toFixed(3)), clip_index: state.currentClipIndex,
  }));
  video.addEventListener("timeupdate", () => {
    if (state.stopAt != null && video.currentTime >= state.stopAt) {
      state.stopAt = null;
      video.pause();
    }
  });
}

function renderExperience() {
  renderProcedureMap();
  renderFocusList();
  const video = document.getElementById("sessionVideo");
  const referenceVideo = document.getElementById("referenceVideo");
  if (state.experience.media?.available) {
    video.src = state.experience.media.url;
    referenceVideo.src = state.experience.media.url;
    video.classList.remove("hidden");
    referenceVideo.classList.remove("hidden");
    document.getElementById("videoUnavailable").classList.add("hidden");
  } else {
    video.classList.add("hidden");
    referenceVideo.classList.add("hidden");
    document.getElementById("videoUnavailable").classList.remove("hidden");
  }
  bindVideo();
  renderExpertMoments();
  const first = feedbackItems()[0];
  if (first) selectFeedback(first.feedback_id, false, "automatic_first_item");
}

function coverageLabel(value) {
  return {
    missed: "Missed by the automated feedback",
    partial: "Partly addressed",
    addressed: "Already addressed",
    unsure: "Coverage unsure",
  }[value] || value;
}

function renderExpertMoments() {
  const root = document.getElementById("expertMomentList");
  if (!state.expertMoments.length) {
    root.innerHTML = `<p class="no-expert-moments">No additional expert coaching moments marked yet.</p>`;
    return;
  }
  root.innerHTML = `
    <h4>Expert-added moments (${state.expertMoments.length})</h4>
    ${state.expertMoments.map(moment => `
      <article class="expert-moment-row">
        <button class="moment-time-button" type="button" data-saved-moment-time="${Number(moment.t_video_s)}">${fmtTime(moment.t_video_s)}</button>
        <div class="expert-moment-content">
          <div class="moment-tags"><span>${escapeHtml(moment.category)}</span><span>${escapeHtml(moment.importance)} priority</span><span>${escapeHtml(coverageLabel(moment.system_coverage))}</span></div>
          ${moment.observation ? `<p><strong>Observation:</strong> ${escapeHtml(moment.observation)}</p>` : ""}
          ${moment.coaching_message ? `<p><strong>Coaching:</strong> ${escapeHtml(moment.coaching_message)}</p>` : ""}
          ${moment.audio?.url ? `<audio controls preload="metadata" src="${escapeHtml(moment.audio.url)}"></audio>` : ""}
        </div>
        <button class="text-button delete-moment-button" type="button" data-delete-moment="${escapeHtml(moment.moment_id)}">Remove</button>
      </article>
    `).join("")}`;
  root.querySelectorAll("[data-saved-moment-time]").forEach(button => button.addEventListener("click", () => {
    const video = document.getElementById("sessionVideo");
    video.pause();
    video.currentTime = Number(button.dataset.savedMomentTime);
    video.scrollIntoView({ behavior: "smooth", block: "center" });
  }));
  root.querySelectorAll("[data-delete-moment]").forEach(button => button.addEventListener("click", async () => {
    if (!window.confirm("Remove this expert-added coaching moment and its local audio recording?")) return;
    button.disabled = true;
    try {
      await api("/api/expert-study/delete-coaching-moment", {
        participant_id: state.participantId,
        moment_id: button.dataset.deleteMoment,
      });
      state.expertMoments = state.expertMoments.filter(row => row.moment_id !== button.dataset.deleteMoment);
      renderExpertMoments();
      renderProcedureMap();
      showToast("Coaching moment removed");
    } catch (error) {
      button.disabled = false;
      showToast(error.message);
    }
  }));
}

function preferredAudioMimeType() {
  if (!window.MediaRecorder) return "";
  return ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus", "audio/webm"]
    .find(type => MediaRecorder.isTypeSupported(type)) || "";
}

function releaseRecorderStream() {
  const stream = state.mediaRecorder?.stream;
  if (stream) stream.getTracks().forEach(track => track.stop());
}

function resetRecording() {
  if (state.mediaRecorder?.state === "recording") {
    state.mediaRecorder.ondataavailable = null;
    state.mediaRecorder.onstop = null;
    state.mediaRecorder.stop();
  }
  releaseRecorderStream();
  state.mediaRecorder = null;
  state.audioChunks = [];
  state.audioBlob = null;
  state.audioDuration = 0;
  state.audioStartedAt = null;
  if (state.audioTimer) window.clearInterval(state.audioTimer);
  state.audioTimer = null;
  if (state.audioPreviewUrl) URL.revokeObjectURL(state.audioPreviewUrl);
  state.audioPreviewUrl = null;
  const preview = document.getElementById("recordingPreview");
  preview.removeAttribute("src");
  preview.classList.add("hidden");
  document.getElementById("recordingStatus").textContent = "Optional · stored only in this local study folder";
  document.getElementById("startRecordingButton").classList.remove("hidden");
  document.getElementById("stopRecordingButton").classList.add("hidden");
  document.getElementById("discardRecordingButton").classList.add("hidden");
}

function openExpertMomentForm() {
  const video = document.getElementById("sessionVideo");
  video.pause();
  state.stopAt = null;
  state.momentTime = Number((video.currentTime || 0).toFixed(3));
  const form = document.getElementById("expertMomentForm");
  form.reset();
  document.getElementById("momentImportance").value = "medium";
  document.getElementById("momentTimeLabel").textContent = fmtTime(state.momentTime);
  document.getElementById("referenceMomentTime").textContent = fmtTime(state.momentTime);
  document.getElementById("momentFormError").classList.add("hidden");
  resetRecording();
  document.getElementById("expertMomentWorkspace").classList.remove("hidden");
  seekReferenceToMarkedFrame();
  document.getElementById("expertMomentWorkspace").scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeExpertMomentForm() {
  resetRecording();
  const referenceVideo = document.getElementById("referenceVideo");
  referenceVideo.pause();
  state.referenceStopAt = null;
  document.getElementById("expertMomentWorkspace").classList.add("hidden");
}

function seekReferenceToMarkedFrame() {
  const video = document.getElementById("referenceVideo");
  state.referenceStopAt = null;
  const seek = () => {
    video.pause();
    const duration = Number.isFinite(video.duration) ? video.duration : state.momentTime;
    video.currentTime = Math.min(Math.max(0, state.momentTime), duration || state.momentTime);
  };
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, { once: true });
}

function replayReferenceMoment() {
  const video = document.getElementById("referenceVideo");
  const duration = Number.isFinite(video.duration)
    ? video.duration
    : Number(state.experience?.media?.duration_s || state.momentTime + 4);
  const start = Math.max(0, state.momentTime - 4);
  state.referenceStopAt = Math.min(duration, state.momentTime + 4);
  const play = () => {
    video.currentTime = start;
    video.play().catch(() => {});
  };
  if (video.readyState >= 1) play();
  else video.addEventListener("loadedmetadata", play, { once: true });
}

async function startRecording() {
  const error = document.getElementById("momentFormError");
  error.classList.add("hidden");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    error.textContent = "This embedded browser does not provide microphone recording. Open the same local study page in Chrome, Edge, or Safari and allow microphone access; typing remains available here.";
    error.classList.remove("hidden");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = preferredAudioMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    state.mediaRecorder = recorder;
    state.audioChunks = [];
    state.audioBlob = null;
    state.audioStartedAt = Date.now();
    recorder.addEventListener("dataavailable", event => {
      if (event.data.size) state.audioChunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      state.audioDuration = Math.min(300, (Date.now() - state.audioStartedAt) / 1000);
      state.audioBlob = new Blob(state.audioChunks, { type: recorder.mimeType || mimeType || "audio/webm" });
      releaseRecorderStream();
      if (state.audioTimer) window.clearInterval(state.audioTimer);
      state.audioTimer = null;
      state.audioPreviewUrl = URL.createObjectURL(state.audioBlob);
      const preview = document.getElementById("recordingPreview");
      preview.src = state.audioPreviewUrl;
      preview.classList.remove("hidden");
      document.getElementById("recordingStatus").textContent = `Recorded ${state.audioDuration.toFixed(1)} seconds · ready to save`;
      document.getElementById("startRecordingButton").classList.add("hidden");
      document.getElementById("stopRecordingButton").classList.add("hidden");
      document.getElementById("discardRecordingButton").classList.remove("hidden");
    }, { once: true });
    recorder.start(1000);
    logEvent("audio_recording_started", { t_video_s: state.momentTime }, null);
    document.getElementById("startRecordingButton").classList.add("hidden");
    document.getElementById("stopRecordingButton").classList.remove("hidden");
    document.getElementById("recordingStatus").textContent = "Recording… 0:00";
    state.audioTimer = window.setInterval(() => {
      const elapsed = (Date.now() - state.audioStartedAt) / 1000;
      document.getElementById("recordingStatus").textContent = `Recording… ${fmtTime(elapsed)}`;
      if (elapsed >= 180 && recorder.state === "recording") recorder.stop();
    }, 250);
  } catch (err) {
    const reason = String(err?.name || "");
    error.textContent = {
      NotAllowedError: "Microphone permission was blocked. Allow microphone access for this localhost page, or open it from the desktop shortcut in Chrome, Edge, or Safari.",
      NotFoundError: "No microphone was detected. Connect or enable a microphone, or continue by typing.",
      NotReadableError: "The microphone is currently unavailable or in use by another application. Close the other application and try again.",
      SecurityError: "This browser blocked microphone access. Open the study from the desktop shortcut in Chrome, Edge, or Safari.",
    }[reason] || "Microphone recording is not available in this browser. Open the study from the desktop shortcut in Chrome, Edge, or Safari, or continue by typing.";
    error.classList.remove("hidden");
  }
}

function stopRecording() {
  if (state.mediaRecorder?.state === "recording") state.mediaRecorder.stop();
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result).split(",", 2)[1] || ""), { once: true });
    reader.addEventListener("error", () => reject(new Error("Unable to read the audio recording")), { once: true });
    reader.readAsDataURL(blob);
  });
}

async function saveExpertMoment(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const error = document.getElementById("momentFormError");
  const data = new FormData(form);
  const observation = String(data.get("observation") || "").trim();
  const coachingMessage = String(data.get("coaching_message") || "").trim();
  if (!observation && !coachingMessage && !state.audioBlob) {
    error.textContent = "Type an observation, coaching message, or record a voice note.";
    error.classList.remove("hidden");
    return;
  }
  if (state.mediaRecorder?.state === "recording") {
    error.textContent = "Stop the recording before saving this coaching moment.";
    error.classList.remove("hidden");
    return;
  }
  button.disabled = true;
  error.classList.add("hidden");
  try {
    const audioBase64 = state.audioBlob ? await blobToBase64(state.audioBlob) : "";
    const result = await api("/api/expert-study/coaching-moment", {
      participant_id: state.participantId,
      t_video_s: state.momentTime,
      category: data.get("category"),
      importance: data.get("importance"),
      system_coverage: data.get("system_coverage"),
      observation,
      coaching_message: coachingMessage,
      audio_base64: audioBase64,
      audio_mime_type: state.audioBlob?.type || "",
      audio_duration_s: state.audioBlob ? state.audioDuration : null,
    });
    state.expertMoments.push(result.moment);
    state.expertMoments.sort((a, b) => Number(a.t_video_s) - Number(b.t_video_s));
    closeExpertMomentForm();
    renderExpertMoments();
    renderProcedureMap();
    showToast("Expert coaching moment saved locally");
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
    button.disabled = false;
  }
}

function scaleQuestion(name, title) {
  return `<div class="question-block"><p class="question-title">${escapeHtml(title)}</p><div class="choice-row scale-row">
    ${[1, 2, 3, 4, 5].map((value, index) => `<label><input type="radio" name="${name}" value="${value}" ${index === 0 ? "required" : ""} /><span>${value}</span></label>`).join("")}
  </div><div class="scale-anchors"><span>Low</span><span>High</span></div></div>`;
}

function openReflection() {
  document.getElementById("experiencePanel").classList.add("hidden");
  document.getElementById("reflectionPanel").classList.remove("hidden");
  setHeaderStatus("Overall reflection");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function returnToExperience() {
  document.getElementById("reflectionPanel").classList.add("hidden");
  document.getElementById("experiencePanel").classList.remove("hidden");
  setHeaderStatus("Feedback ready");
}

async function submitReflection(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const error = document.getElementById("reflectionError");
  button.disabled = true;
  error.classList.add("hidden");
  try {
    const data = new FormData(form);
    const answers = Object.fromEntries([
      "workflow_clarity", "feedback_usefulness", "debrief_fit", "would_use",
      "most_useful", "missing_or_concerning",
    ].map(field => [field, data.get(field) || ""]));
    const result = await api("/api/expert-study/complete-experience", {
      participant_id: state.participantId,
      answers,
    });
    state.start.case_completed = true;
    state.start.case.next_case_url = result.next_case_url;
    state.start.completion_code = result.completion_code;
    document.getElementById("reflectionPanel").classList.add("hidden");
    showCompletion();
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
    button.disabled = false;
  }
}

function showCompletion() {
  ["accessPanel", "readyPanel", "processingPanel", "experiencePanel", "reflectionPanel"].forEach(id => {
    document.getElementById(id).classList.add("hidden");
  });
  document.getElementById("completionPanel").classList.remove("hidden");
  setHeaderStatus("Case complete");
  const nextButton = document.getElementById("nextCaseButton");
  const codeBlock = document.getElementById("completionCodeBlock");
  if (state.start.case.next_case_url) {
    document.getElementById("completionMessage").textContent = "Your workflow reflection was saved. Continue when you are ready to experience the next recorded session.";
    nextButton.classList.remove("hidden");
    codeBlock.classList.add("hidden");
  } else {
    document.getElementById("completionTitle").textContent = "Expert workflow experience complete";
    document.getElementById("completionMessage").textContent = "All configured cases are complete. Please provide the completion code to the study team.";
    nextButton.classList.add("hidden");
    document.getElementById("completionCode").textContent = state.start.completion_code || "Complete";
    codeBlock.classList.remove("hidden");
  }
}

function goToNextCase() {
  const next = new URL(state.start.case.next_case_url, window.location.href);
  next.searchParams.set("participant_id", state.participantId);
  logEvent("next_case_opened", { next_case_index: state.start.case.index + 1 }, null);
  window.location.assign(next.toString());
}

document.getElementById("reflectionScales").innerHTML = [
  scaleQuestion("workflow_clarity", "How clear was the end-to-end workflow?"),
  scaleQuestion("feedback_usefulness", "How useful was the resulting feedback?"),
  scaleQuestion("debrief_fit", "How well could this fit a simulator debrief?")
].join("");

document.getElementById("accessForm").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  const error = document.getElementById("accessError");
  button.disabled = true;
  error.classList.add("hidden");
  try {
    const participantId = document.getElementById("participantId").value.trim();
    const firstCaseUrl = await firstCaseUrlForAccess(participantId);
    if (firstCaseUrl) {
      window.location.assign(firstCaseUrl);
      return;
    }
    await enterStudy(participantId);
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
    button.disabled = false;
  }
});
document.getElementById("startAnalysisButton").addEventListener("click", startAnalysis);
document.getElementById("finishExperienceButton").addEventListener("click", openReflection);
document.getElementById("returnToExperienceButton").addEventListener("click", returnToExperience);
document.getElementById("reflectionForm").addEventListener("submit", submitReflection);
document.getElementById("nextCaseButton").addEventListener("click", goToNextCase);
document.getElementById("openMomentButton").addEventListener("click", openExpertMomentForm);
document.getElementById("cancelMomentButton").addEventListener("click", closeExpertMomentForm);
document.getElementById("startRecordingButton").addEventListener("click", startRecording);
document.getElementById("stopRecordingButton").addEventListener("click", stopRecording);
document.getElementById("discardRecordingButton").addEventListener("click", resetRecording);
document.getElementById("expertMomentForm").addEventListener("submit", saveExpertMoment);
document.getElementById("replayReferenceButton").addEventListener("click", replayReferenceMoment);
document.getElementById("returnToMarkedFrameButton").addEventListener("click", seekReferenceToMarkedFrame);
document.getElementById("referenceVideo").addEventListener("timeupdate", event => {
  if (state.referenceStopAt != null && event.currentTarget.currentTime >= state.referenceStopAt) {
    state.referenceStopAt = null;
    event.currentTarget.pause();
    event.currentTarget.currentTime = state.momentTime;
  }
});
document.getElementById("supportDetails").addEventListener("toggle", event => {
  if (event.currentTarget.open) logEvent("support_details_opened");
});

const participantFromUrl = new URL(window.location.href).searchParams.get("participant_id");
if (participantFromUrl) {
  document.getElementById("participantId").value = participantFromUrl;
  document.getElementById("studyReady").checked = true;
  enterStudy(participantFromUrl).catch(err => {
    const error = document.getElementById("accessError");
    error.textContent = err.message;
    error.classList.remove("hidden");
  });
}
