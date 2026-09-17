# Task tabs UI plan

Status: the initial task shell was implemented 2026-09-09; the end-to-end
workflow was extended 2026-09-10. The current UI is documented in the README's
app workflow section. The detailed original contract below remains useful for
Paint controls, draft handling, exact scope, shortcuts and candidate safety.

Current changes superseding the original navigation below:

- Workspace tasks are Run → Inspect → Paint → Compare → Export. Internal task
  IDs `rerun` and `review` remain compatible with stored context.
- Import and Open are separate centered panels. Import assumes `/img_nir`,
  prepares its flat field with loading status, and explicitly chooses the entire
  recording or selected bounds before workspace creation.
- Statistics, Layers, Jobs and History share the right panel's tabs. Shortcuts
  is a toggleable drawer. Training can open Jobs; other non-workspace screens
  hide frame-specific diagnostics and restore the workspace pane on return.
- Run guides first-time setup and checked stage execution; export is separate.
- Inspect persists human review independently from automatic flags and lists
  attention segments, with an adjustable minimum length of 8 sampled frames by
  default. Filtering affects the queue and next/skip navigation, not flags or
  saved review. Corrections invalidate affected frames and their neighbors,
  preserving reviews of unchanged segments. Global input/configuration changes
  invalidate all reviews.
- Paint can save the current mask and prepare a selected-range refit.
- Compare supports Current/A/B overlays and synchronized central previews.
- Export captures named Parquet from a matching snapshot with configuration,
  masks, provenance and review records under the workspace writer lock.
- The usability pass adds a shared frame/range/draft context strip, explicit
  Open/Resume actions, keyboard recording selection, inline recoverable errors,
  checkpoint availability checks, pinned Paint save actions, one comparison
  acceptance group, confirmed option deletion, and a central Export checklist.

## Layout and navigation

- Header: active workspace and recording, Import/Open, Labels, Training.
  Opening work is a dedicated library screen, not a permanent list above tools.
- Left panel: Inspect / Paint / Rerun / Review, followed by only that task's
  controls. Approximately 320 px, resizable. Paint has permanently expanded
  Brush, Proposals, Refine and draft actions, followed by Next frame and
  Save & refit in the same scrollable left panel. The canvas has no separate
  bottom Paint tray; the timeline is the only bottom panel. No Paint sub-tabs, accordions,
  hover-only actions or hidden advanced controls. Inapplicable controls remain
  visible and disabled with a reason. Narrow windows reflow and can scroll;
  the left panel scrolls independently on desktop without overlapping controls.
- Center: video canvas. Raw/flat, overlay visibility/opacity, fit view, layers,
  and optional statistics belong beside this canvas and persist across tasks.
- Bottom: frame transport, exact frame input, FPS, selection bounds, loop
  selection, timeline. Extra series and provenance tracks can expand on demand.
- Jobs and History open temporary drawers. Statistics, hypotheses, width,
  curvature, mask statistics and summary use a collapsible inspector.
- Preserve workspace, playhead, temporal selection, zoom, pan, layers and each
  task's form values when changing tabs. Pause playback upon entering Paint.
- A playhead is not an operation scope: viewing frame 128 must not silently
  change a selected region of 120–160. Paint always edits one frame. Rerun and
  candidate acceptance display exact bounds and frame counts.

## Detailed screen states

1. **Open work:** two clear paths, Import recording and Open workspace. Import
   previews the file, allows HDF5 dataset selection, first/last/step and a name,
   then creates a workspace. Existing runs remain importable as workspace copies.
   Recent workspaces show recording, range and last edited time. Restore the
   saved viewing context on open. General video formats are separate backend work;
   the existing HDF5 path remains the initial supported import flow.
2. **Inspect:** seek flagged / low-IoU / stretch / pose-jump / edge / provenance
   frames, choose a worst frame, set thresholds and inspect notes. Select an
   individual frame or range from the timeline or exact bounds. Display layers,
   fitter starts, comparison workspace and diagnostics remain available.
3. **Paint:** visible target banner (Workspace mask or Corpus label), with all
   brush, proposal, refinement, navigation and save controls expanded together.
   Source selection changes the proposal preview, not the available controls.
   Mask proposals are distinct from pose candidates in Review. Network threshold,
   Saved-source choice, stride, uncertainty candidate count, manifest selection,
   split pledge and save destinations retain their layout positions even when
   disabled. Clear draft, Discard draft, Remove override and Delete corpus label
   are separate, visible actions with explicit targets.
4. **Rerun:** first choose current frame, selected range or whole workspace.
   Regional work uses the six existing algorithms: independent multi-start,
   slow refit, forward chain, backward chain, beam path, mirror. Show anchors,
   method-specific parameters, input mask source and explicit Run action.
   Whole-workspace mode retains stage selection, stage forms and Run checked
   stages in order. Do not silently widen scope for unsupported operations.
   Arbitrary-range segmentation or additional stages require explicit backend
   support before appearing as available range actions.
5. **Review:** choose completed candidate set(s), compare current/A/B using
   overlays or side by side, inspect metrics and acceptance bounds, explicitly
   Accept or Keep current. Retain frame hypothesis picks and orientation flips.
   Changed mask inputs disable acceptance and offer a new run. Keep outcome
   history reachable without mixing it into the run form.
6. **Labels:** full corpus browser with structured source/split/recording
   filters, counts, revision, source path, split and deletion. Open a saved label
   in the same Paint tools with a prominent Corpus label target. Label new frames
   from a declared recording pool or manifest without implicitly creating or
   mutating workspaces. Returning to the workspace restores its viewing context.
7. **Training:** corpus counts, training settings, jobs and completed checkpoints.
   Retain crop, patience, workers, seed, epochs, batch size, learning rate and
   device. Keep explicit checkpoint selection for the active workspace. Training
   controls do not appear among frame-correction tools.

## Labeller controls to restore

| Original control | Current integration | Planned home and behavior |
|---|---|---|
| Network / Classical / raw Threshold / Saved proposals | Missing as draft inputs | Paint → Propose, with separate network threshold and proposal preview |
| Replace / union / intersect / subtract | Missing | Explicit combine selector; preserve Shift / Alt / Ctrl-or-Cmd proposal modifiers |
| Network threshold → apply proposal | Viewer threshold exists but is not an editable draft proposal | Paint → Propose; slider changes preview only, Apply changes draft |
| Fill holes / largest component / grow / shrink | Missing | Paint → Refine; each operation is undoable |
| Tube fit | Missing as mask refinement | Paint → Refine; creates a draft mask, does not accept a pose |
| Save + next | Missing | Paint save area; advances only after all selected saves succeed |
| Queue (manifest) | Only legacy launcher supports it | Paint → Next frame, manifest source and progress; preserve manifest split pledges |
| Network-uncertain | Missing | Paint → Next frame, candidate count and selected source pool |
| Random unlabeled | Missing | Paint → Next frame, selected source pool |
| Sequential stride | Single-frame stepping remains, not labelling stride | Paint → Next frame, explicit stride |
| Browse filtered saved labels | Can open labels individually, no traversal | Labels filters + Paint → Next frame; preserve browse cursor when a save changes filter membership |
| Previous visited frame | Only temporal stepping remains | Paint Previous / P uses labeling visit history; timeline arrows still step frames |
| Clear mask to background | Clear override has different semantics | Paint → Draw, Clear draft; undoable. Remove override is a distinct saved operation |
| Overlay opacity cycle | Basic visibility remains | Canvas display controls; restore 45% / 20% / off cycle with O |
| Source / split / current-recording filters | Only free-text corpus filter | Labels browser, also used by Browse saved labels mode |
| Brush/refine/proposal/navigation shortcuts | Most are missing or conflict with viewer actions | Globally unique bindings, described below; no task-based reassignment |
| Right/middle or Shift-drag pan | Alt/middle pan currently used in Paint | Restore old gestures; retain Alt-drag as an alias where it does not conflict |

Retain existing worm/background/ignore brushes, size control, dirty protection,
saved-mask undo, raw/flat display, zoom/fit, corpus delete, split pledges, corpus
revisions and training. Restoring controls must not remove these improvements.

## Save, draft and navigation behavior

- Workspace mode defaults to **Save mask**, with a clearly labeled optional
  **Also save a training label** checkbox, off initially. Keep the split control
  visible but disabled when inapplicable; honor existing and manifest pledges.
- **Save + next** performs those same selected saves and then follows Next mode.
  Labels mode saves the corpus copy only, matching the original labeller.
- If a workspace save succeeds but a corpus save fails, remain on the frame,
  report the two outcomes separately and retry only the incomplete destination.
  Never advance on a failed save or obscure which copy was written.
- Proposal application, refinement, painting and Clear draft all enter the same
  per-frame draft undo stack. Discard draft restores the last saved target.
  Saved proposal names its source explicitly: saved workspace override or saved
  corpus label. The automatic mask and independent corpus copy remain distinct
  inputs; the original corpus Saved proposal must still be accessible from a
  workspace when a matching label exists.
  Remove override restores the automatic workspace mask and is undoable through
  saved edit history. It is not the same action as Clear draft.
- Saving a workspace mask makes affected poses stale. Offer Refit this frame or
  Refit selected range, taking the user to Rerun with scope filled in; never
  auto-accept a pose. Saving a corpus label does not stale workspace poses.
- Tab changes preserve drafts. Navigating away from their frame, target or
  workspace uses Save and continue / Discard / Stay. Jobs cannot consume an
  unsaved draft; Rerun offers Save and use mask or Return to Paint.
- New-frame traversal stays within the declared pool: current workspace/selection
  in correction mode, selected recordings/manifest in corpus labeling mode.
  Queue or uncertainty traversal across recordings must not silently switch the
  active correction workspace. Missing source recordings show an actionable state.
- No network checkpoint: disable Network proposal and Network-uncertain mode;
  keep Classical, Threshold, Saved and brush editing available. Missing saved
  label disables Saved. Empty filters/queue report completion without looping
  unexpectedly. Painting and saving remain disabled for read-only runs.

## Keyboard and pointer behavior

One exact keyboard chord has one action throughout the application. Task and
target determine whether that action is enabled, never what it means. A single
registry drives dispatch, visible key labels and help; initialization and tests
reject duplicate normalized chords. Feature modules register actions, not global
keyboard listeners. No legacy shortcut aliases that reintroduce collisions.

| Key / chord | Only meaning |
|---|---|
| Space | Play/pause; disabled while actively editing a Paint draft |
| Left / Right; Shift+arrows; Ctrl+arrows | Temporal step 1 / 10 / 100 frames, through the draft navigation guard |
| `[` / `]` | Previous / next flagged frame |
| `,` / `.` | Previous / next low-IoU frame |
| `1`–`9` | Display layer toggles |
| F / O / 0 | Raw/flat / overlay opacity cycle / fit view |
| M / K | Notes / fitter starts, moved from N / S |
| B / E / I | Worm / background / ignore brush |
| `-` / `=` | Smaller / larger brush |
| W / C / T / V | Preview Network / Classical / raw Threshold / Saved proposal |
| A | Apply selected proposal using the visible combine mode |
| H / L / D / R / U | Fill holes / largest component / grow / shrink / Tube fit |
| N / P | Next labeling target / previous visited labeling target |
| S | Save the Paint target to the selected destinations |
| Enter | Save + next |
| Z; Ctrl+Z or Cmd+Z | Undo draft only; never falls through to a saved edit |

Undo saved operations and destructive actions use explicit buttons. Do not assign
proposal modifier accelerators such as Ctrl+W/T/C/V, which conflict with browser
or clipboard behavior. Shift/Alt/Ctrl-or-Cmd clicks on proposal buttons may apply
union/intersect/subtract; the combine selector exposes these operations without
modifiers. Exact modifier matching prevents plain-letter handlers from consuming
browser shortcuts. Suppress app accelerators in inputs/selects/textareas,
contenteditable and dialogs, and preserve native Enter/Space on focused buttons.
Auto-repeat is allowed only for stepping and brush sizing, not saving or editing.
Normalize shifted letters and platform modifiers centrally.

Paint only on an unmodified primary-button drag. Right/middle/Shift/Alt canvas
drag pans; Shift-drag on the timeline selects a range. Pointer capture must not
let these gestures start a paint stroke. Keep each gesture local to its surface.

## Parallel implementation plan

Use **three Astra agents at medium reasoning**, with the primary agent integrating
the shell. The planning agents have completed read-only audits; implementation
starts only after this plan is accepted. Do not let workers edit shared layout or
bootstrap files concurrently.

### Gate 0: freeze interfaces and ownership

The integrator first records the DOM mount points, exported action names,
normalized shortcut registry and API schemas. Workers may use fixtures against
these contracts immediately; backend completion does not gate UI authoring.

- `PaintTarget`: kind (`workspace`, `corpus`, or new corpus frame), canonical
  recording path + HDF5 dataset + source frame, workspace/sample identity,
  independent expected workspace/corpus revisions, and checkpoint identity.
- `DraftSnapshot`: target, generation, draft revision, encoded labels, saved
  baseline, undo stack and frozen save destinations. Async computations return
  target/generation/revision; stale completions cannot replace newer painting.
- `Selection`: preserve the existing row-index region representation; derive
  displayed/API frames through `series.frame_index`. Rerun scope is a separate
  `current | selection | workspace` value. Current-frame refit must not replace
  the user's selected range. Cover stepped and noncontiguous recordings.
- `Navigation.requestLeave(destination)`: asynchronous Save and continue /
  Discard / Stay. Tab switches do not leave the draft. All frame/source/target
  navigation uses this one guard. Entering corpus Paint snapshots workspace
  viewing context; returning restores it.
- `SaveResult`: per-destination success/revision/error for one immutable draft
  snapshot. Save + next advances only when every requested destination succeeds.
  Retry skips already-successful destinations for that same snapshot.
- Workers expose `mount`, `refresh`, state snapshot/restore, and named action
  handlers through agreed adapters. They deliver DOM/CSS requirements to the
  integrator rather than independently restructuring `index.html`.

### Wave 1: concurrent feature work

| Owner | Files owned | Deliverable |
|---|---|---|
| Primary integrator | UI `index.html`, `style.css`, `app.js`, `api.js`, `panels.js`, `viewer.js`, `charts.js`; new task-state/shortcut module; layout tests | Header/screens, Inspect/Paint/Rerun/Review order, expanded Paint layout, shared state and draft guard integration, one shortcut dispatcher, transport/selection, stage forms, Jobs/History and inspector |
| Astra A — labeling backend | New `app/labeling.py`, `app/routers/labeling.py`; `corpus.py`, corpus router; Python `app/state.py`, route registration, legacy launcher adaptation; Python labeling/corpus tests | Proposal/refinement APIs, canonical target resolution, next modes and manifest handling, structured corpus filters, optimistic saves and explicit exhaustion |
| Astra B — Paint frontend | `pose_viewer_ui/masks.js`; new `paint_navigation.js` and optional draft/proposal module; dedicated Paint browser tests | All brush/proposal/refine controls, one draft undo stack, stale async guards, frozen multi-destination saves, Save + next, navigation modes/visit history, safe pan/paint routing |
| Astra C — Labels and Review frontend | `pose_viewer_ui/corpus.js`, `regions.js`, `edits.js`; optional `review.js`; dedicated labels/review browser tests | Corpus filtering/editing/new-label entry, Training screen adapters, rerun/review separation, candidate comparison/acceptance, explicit saved undo, corpus return context |

No worker edits another owner's files. Cross-file changes are requested as small
integration patches/messages. Shared test fixtures and script includes are owned
by the integrator; feature tests use separate files. API or DOM contract changes
are agreed before changing consumers. Existing user modifications are preserved.

### Backend contract baseline

These are proposed new endpoints, not claims about implemented APIs:

| Endpoint | Contract |
|---|---|
| POST `/api/labeling/frame` | Explicit workspace/corpus/new-recording target → canonical target, images/labels, dimensions, independent revisions, split pledge and feature availability |
| POST `/api/labeling/proposals` | Target + source + request ID → source-labeled mask/probability, checkpoint identity and echoed target/request ID; no writes |
| POST `/api/labeling/refine` | Target + draft labels + method + generation/revision → refined labels/metrics and echoed identity; no writes or accepted pose |
| POST `/api/labeling/manifests` | Manifest path → validated canonical entries, identity, pledges, progress and missing-source diagnostics |
| POST `/api/labeling/next` | Explicit pool + mode + cursor + stride/candidate count/filters → next target, cursor, progress, or explicit exhausted response |
| GET `/api/corpus` extensions | Source/split/canonical-recording filters; stable ordering, filtered counts and facets while preserving existing response fields |
| POST `/api/corpus/labels` extensions | Direct target + frozen labels + expected corpus revision/absence; optional split/manifest. Retain legacy workspace-copy requests; new UI supplies expected workspace mask revision when using that path |

Reuse codecs, recording sources, segmentation caches, `Proposer.classical` and
`Proposer.refine`; keep threshold/combine draft operations local. Do not reuse
legacy `LabelState` wholesale: basename identities, modulo wrapping, bounded random
fallback and silent uncertainty fallback are incompatible with this plan. Reuse
manifest semantics through canonical path/dataset/frame identities. Existing
split pledges win, including after deletion. Detect exhaustion explicitly and
sample distinct uncertainty candidates without widening the declared pool.

For saved corpus labels, stored images allow editing even when source recordings
are unavailable. Source-dependent proposal operations can be unavailable without
blocking brush edits. Enforce revision/expected-absence checks inside locks.
Freeze the corpus save's label bytes rather than rereading a possibly changed
workspace override. Keep workspace → corpus lock order and avoid holding corpus
locks through inference or reacquiring the same store lock recursively.

Preserve label encoding and existing ignore-pixel semantics in tested fixtures;
do not treat internal value 255 as worm. No whole-pipeline rewrites, generic video
import or new arbitrary-range stages are included. Existing mask staleness,
candidate fingerprints, region acceptance and historical provenance remain intact.

### Wave 2: integrate and validate

The primary agent integrates in dependency order: shell mounts and action registry,
backend routes, Paint adapters, Labels/Review adapters, then legacy-launcher routing.
Once a feature agent finishes, reuse that Astra medium slot for independent
acceptance testing against the assembled app; remaining agents resolve findings
only in their owned files. Do not start a fourth concurrent subagent.

Completion gates:

1. Every Paint control is rendered without sub-tabs/disclosures. Verify desktop
   layouts at 1280×800, 1440×900 and 1920×1080, plus narrow width and 200% zoom.
   Controls never overlap; narrow layouts reflow/scroll without hiding groups.
2. Shortcut registry contains no duplicate normalized chords. Test every binding
   across all tasks: at most one action, unchanged meaning, inactive when
   unavailable, no input/dialog leakage, no saved-undo fallback, and no repeated
   saves from held keys. Help and button labels match the registry.
3. Brush/proposal/refine/Clear share draft undo; threshold preview is nonmutating;
   ignore labels survive the defined operations; pan gestures never paint.
4. All five next modes honor pool, cursor, exhaustion and pledges. Failed or
   partially successful saves do not advance or duplicate successful writes.
5. Tab switches preserve playhead, range, zoom and drafts. Corpus return restores
   the workspace. Current-frame reruns preserve range selection. Late computation
   results cannot overwrite a different target or newer draft.
6. End-to-end: import/open → Inspect → Paint → save → Rerun → Review → accept →
   saved undo; also corpus-only labeling, queue traversal, training/checkpoint
   selection, read-only behavior and missing checkpoint/source states.
7. Focused unit/API/browser checks first, then the existing affected app, mask,
   corpus, pipeline and region regression suites once the integrated build is stable.

## Preserved acceptance checklist

1. Introduce task shell and state ownership; move existing controls with a
   preservation checklist. Verify frame, range, zoom and drafts survive tabs.
2. Restore draft proposal/refinement behavior and APIs, reusing the original
   segmenter/refinement helpers behind unified app routes. Verify undo and ignore
   labels for each operation; handle slow Tube fit without blocking navigation.
3. Restore navigation modes, manifest identity/pledges, corpus filters, visit
   history and Save + next. Verify successful, failed and partially failed saves,
   exhaustion, cross-recording corpus work and filter changes after relabeling.
4. Separate Review, Labels, Training and drawers; preserve existing candidate
   fingerprints, acceptance, history, checkpoints and provenance semantics.
5. Browser acceptance: import/open → inspect/select → apply proposal/refine/paint
   → save → scoped refit → compare/accept → undo. Also exercise every restored
   next mode, all shortcuts, corpus-only editing, reload/draft protection and
   read-only state. Update launcher documentation after restoration is complete.

This pulls forward only the manifest labeling workflow from the legacy app.
It does not require implementing the broader Phase 6 severity/audit queue.

## Implementation verification

The shared shell, Paint/navigation controllers, labeling API and Labels/Review
adapters implement this contract. Import remains HDF5-based as scoped above.

- Chromium regressions cover restored Paint controls and modifiers, proposal and
  refinement races, frozen dual-save retries, all five traversal modes, startup
  manifests, corpus return context, independent rerun scope and stale candidates.
- The shortcut regression checks all 47 canonical chords across all four tasks,
  input/dialog focus, native button behavior, repeated keys and draft-only undo.
- The complete synthetic browser workflow paints, edits corpus revisions, trains
  one CPU epoch, selects the checkpoint, refits a frame, accepts it and undoes the
  saved edit. A real-backend startup manifest test opens a corpus frame with its
  pledged split and returns to the previous workspace without writing a label.
- Desktop layouts at 1280×800, 1440×900 and 1920×1080 and narrow/zoom-equivalent
  layouts at 720×450 and 390×844 have no horizontal overflow or hidden control
  groups. After the layout revision, the expanded Paint sidebar scrolls vertically
  to reach Next frame and Save & refit, leaving more height for the canvas.
- Focused Python checks cover labeling API semantics, corpus revisions, launcher
  compatibility and shared app/read-only-viewer static assets.
- An additional 58-test CPU regression run completed 56 passes and one failure
  marker before being interrupted during the final, lengthy pipeline fit test.
  The failure position matches the static-page title assertion; both app and
  read-only viewer static tests pass after updating the expected title. The
  interrupted run did not emit its stored failure traceback and is not counted
  as a complete suite pass.
