"use strict";

// Inspection owns human review; automatic flags continue to describe fit quality.
const workflowInspect = (() => {
  let payload = null, source = null, request = 0, initialized = false;
  const minimumKey = 'poseViewer.inspectionMinimumFrames';
  let minimumFrames = 8;
  const eligible = (segments = []) => segments.filter(s => s.last - s.first + 1 >= minimumFrames);
  const node = id => document.getElementById(id);
  const supported = () => serverIsApp() && isWorkspace() && !!state.run;
  const url = () => `/api/workspaces/${encodeURIComponent(state.runName)}/inspection`;
  const selected = () => payload?.segments.find(s => state.region?.first === s.first && state.region?.last === s.last);
  const message = text => { if (node('inspection-status')) node('inspection-status').textContent = text; };
  function render() {
    if (!initialized) return;
    const summary = payload?.summary, segment = selected();
    node('inspection-summary').textContent = summary ? `${summary.unprocessed_frames} unprocessed frames · ${summary.flagged_frames} automatically flagged · ${summary.reviewed_frames} human reviewed` : supported() ? 'Loading inspection…' : 'Open a workspace to inspect and record human review.';
    node('inspection-target').textContent = state.region && state.run ? `Selected frames ${frameOfRow(state.region.first)}–${frameOfRow(state.region.last)}${segment?.reviewed ? ' · human reviewed' : ''}` : 'Select a segment needing attention, or select a range on the timeline.';
    const list = node('inspection-segments'); list.replaceChildren();
    for (const s of eligible(payload?.segments)) {
      if (s.reviewed && !node('inspection-show-reviewed').checked) continue;
      const button = document.createElement('button'); button.type = 'button'; button.className = 'inspection-segment';
      button.setAttribute('aria-pressed', String(s === segment));
      const range = document.createElement('strong'), detail = document.createElement('span');
      range.textContent = `Frames ${s.frames[0]}–${s.frames[1]} · ${s.last - s.first + 1} ${s.last === s.first ? 'frame' : 'frames'}`;
      detail.textContent = `${(s.reasons || []).join(', ').replaceAll('_', ' ') || 'No fitted pose'} · ${s.reviewed ? 'Human reviewed' : 'Needs human review'}`;
      button.append(range, detail);
      button.onclick = () => select(s); list.append(button);
    }
    if (payload && !list.children.length) {
      const empty = document.createElement('p'); empty.className = 'note';
      const shorter = payload.segments.some(s => (!s.reviewed || node('inspection-show-reviewed').checked) && s.last - s.first + 1 < minimumFrames);
      empty.textContent = shorter ? `No segments match the minimum of ${minimumFrames} frames. Shorter segments are hidden.` : 'No segments need review. Automatic flags remain visible after review.';
      list.append(empty);
      if (shorter) {
        const show = document.createElement('button'); show.type = 'button'; show.id = 'inspection-show-shorter';
        show.textContent = 'Show shorter segments'; show.onclick = () => setMinimum(1); list.append(show);
      }
    }
    for (const id of ['inspection-paint','inspection-orientation','inspection-fit','inspection-mark']) node(id).disabled = !supported() || !state.region || !payload;
    node('inspection-next').disabled = !eligible(payload?.segments).some(s => !s.reviewed);
    node('inspection-loop').checked = !!node('loop-stretch')?.checked;
  }
  function setMinimum(value) {
    minimumFrames = value; node('inspection-min-frames').value = value;
    node('inspection-min-frames').setCustomValidity('');
    node('inspection-min-frames').removeAttribute('aria-invalid');
    node('inspection-min-error').textContent = '';
    writeStorage(minimumKey, value); render();
  }
  async function refresh() {
    if (!initialized) return null;
    const key = currentSourceKey(), token = ++request;
    if (key !== source) { payload = null; source = key; }
    if (!supported()) { payload = null; render(); return null; }
    try { const result = await api(url()); if (token !== request || key !== currentSourceKey()) return null; payload = result; message('Human review does not clear automatic flags. Changed frames and their neighbors need review again; reviews of unchanged segments are preserved.'); render(); return result; }
    catch (error) { if (token === request) { payload = null; render(); message(error.message); } return null; }
  }
  async function select(segment) {
    const key = currentSourceKey();
    if (!await maskEditor.requestLeave({destination:{kind:'frame',row:segment.first}}) || key !== currentSourceKey()) return false;
    if (state.playing) togglePlay();
    setRegionRows(segment.first, segment.last, `${segment.kind} segment needing inspection`);
    await showRow(segment.first, {keepView:true,skipMaskGuard:true});
    syncTaskSelection(); render(); return true;
  }
  async function next() {
    const fresh = await refresh(); if (!fresh) return false;
    const pending = eligible(fresh.segments).filter(s => !s.reviewed);
    const segment = pending.find(s => s.first > (state.region?.last ?? state.row)) || pending[0];
    if (!segment) { message(fresh.segments.some(s => !s.reviewed) ? `No unreviewed segments meet the minimum of ${minimumFrames} frames. Lower the minimum to review shorter segments.` : 'All attention segments have been reviewed.'); return false; }
    if (!await select(segment)) return false;
    showTab('inspect'); return true;
  }
  async function mark() {
    if (!supported() || !state.region || !payload) return;
    const key = currentSourceKey(), bounds = {...state.region}, revision = payload.revision;
    if (!await maskEditor.requestLeave({preserveTarget:true}) || key !== currentSourceKey()) return;
    if (state.region?.first !== bounds.first || state.region?.last !== bounds.last) { message('Selection changed. Review the selected range before marking it reviewed.'); return; }
    try { const result = await post(`${url()}/review`, {first:bounds.first,last:bounds.last,revision}); if (key !== currentSourceKey()) return; payload = result; render(); message('Selected range marked human reviewed. Automatic flags are unchanged.'); window.dispatchEvent(new CustomEvent('workflow:review')); await next(); }
    catch (error) { message(error.message); }
  }
  async function orientation() {
    if (!state.region || !maskEditor.beforeMutation()) return;
    const frames = state.run.series.frame_index.slice(state.region.first, state.region.last + 1);
    await submitEdit({kind:'flip',frame:frames[0],frames}, `Corrected orientation on frames ${frames[0]}–${frames.at(-1)}`);
    await refresh();
  }
  function init() {
    if (initialized) return; initialized = true;
    const section = document.createElement('section'); section.className = 'group inspection-workflow';
    section.innerHTML = `<h3>Segments needing attention</h3><p id="inspection-summary" class="note" aria-live="polite"></p><label for="inspection-min-frames">Minimum segment length (frames)</label><input id="inspection-min-frames" type="number" min="1" step="1" value="8" aria-describedby="inspection-min-help inspection-min-error"><p id="inspection-min-help" class="note">Counts sampled frames in each segment. Applies to this queue and next-segment navigation.</p><p id="inspection-min-error" class="note" aria-live="polite"></p><label><input id="inspection-show-reviewed" type="checkbox"> Include human reviewed segments</label><div id="inspection-segments" class="inspection-segments" aria-label="Segments needing attention"></div><p id="inspection-target" class="target-banner"></p><label><input id="inspection-loop" type="checkbox"> Loop selected range during playback</label><div class="inspection-actions"><button id="inspection-paint">Paint mask</button><button id="inspection-orientation">Correct orientation</button><button id="inspection-fit">Try another fit</button><button id="inspection-mark" class="primary">Mark reviewed &amp; next</button><button id="inspection-next">Skip to next unreviewed</button></div><p id="inspection-status" class="note" aria-live="polite"></p>`;
    node('tab-inspect').prepend(section);
    const savedMinimum = readStorage(minimumKey, 8);
    minimumFrames = Number.isSafeInteger(savedMinimum) && savedMinimum >= 1 ? savedMinimum : 8;
    node('inspection-min-frames').value = minimumFrames;
    node('inspection-min-frames').oninput = event => {
      const value = Number(event.target.value);
      if (!Number.isSafeInteger(value) || value < 1) {
        const error = 'Enter a whole number of frames, at least 1.';
        event.target.setCustomValidity(error); event.target.setAttribute('aria-invalid', 'true');
        node('inspection-min-error').textContent = error; return;
      }
      setMinimum(value);
    };
    node('inspection-show-reviewed').onchange = render;
    node('inspection-loop').onchange = event => { if (node('loop-stretch')) node('loop-stretch').checked = event.target.checked; };
    node('inspection-next').onclick = next; node('inspection-mark').onclick = mark;
    node('inspection-paint').onclick = () => showTab('paint');
    node('inspection-fit').onclick = () => { setRerunScope('selection'); showTab('rerun'); };
    node('inspection-orientation').onclick = orientation;
    for (const event of ['workflow:source','workflow:changed']) window.addEventListener(event, refresh);
    window.addEventListener('workflow:task', event => { if (event.detail?.task === 'inspect') refresh(); });
    window.addEventListener('workflow:selection', render);
    render();
  }
  return {init,refresh,sourceChanged:refresh,next,summary:async () => (await refresh())?.summary || null};
})();
window.workflowInspect = workflowInspect;
