# Task tabs UI plan

Status: proposed, 2026-09-09. This is a design and restoration plan; no application
changes are made by this document. Task tabs were selected from the five layout
proposals. The original labeller was audited directly in
`src/worm_pose_gen/label_app_ui/index.html`, `app.js`, and `label_app.py`.

## Layout and navigation

- Header: active workspace and recording, Import/Open, Labels, Training.
  Opening work is a dedicated library screen, not a permanent list above tools.
- Left panel: Inspect / Rerun / Paint / Review, followed by only that task's
  controls. Approximately 320 px, resizable. Keep task navigation and the primary
  action visible; use a single scrollable tool area in the production app.
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
3. **Rerun:** first choose current frame, selected range or whole workspace.
   Regional work uses the six existing algorithms: independent multi-start,
   slow refit, forward chain, backward chain, beam path, mirror. Show anchors,
   method-specific parameters, input mask source and explicit Run action.
   Whole-workspace mode retains stage selection, stage forms and Run checked
   stages in order. Do not silently widen scope for unsupported operations.
   Arbitrary-range segmentation or additional stages require explicit backend
   support before appearing as available range actions.
4. **Paint:** visible target banner (Workspace mask or Corpus label), brush
   controls and a compact Draw / Propose / Refine switch. Next-frame controls
   live in a separate expandable section. The save area remains visible.
   The proposal view shows proposal versus current draft before application;
   mask proposals are distinct from pose candidates in Review.
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
| Brush/refine/proposal/navigation shortcuts | Most are missing or conflict with viewer actions | Task-scoped bindings, described below |
| Right/middle or Shift-drag pan | Alt/middle pan currently used in Paint | Restore old gestures; retain Alt-drag as an alias where it does not conflict |

Retain existing worm/background/ignore brushes, size control, dirty protection,
saved-mask undo, raw/flat display, zoom/fit, corpus delete, split pledges, corpus
revisions and training. Restoring controls must not remove these improvements.

## Save, draft and navigation behavior

- Workspace mode defaults to **Save mask**, with a clearly labeled optional
  **Also save a training label** checkbox, off initially. Show split controls only
  when that destination is selected; honor existing and manifest split pledges.
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

Bindings apply only when the canvas/task has focus, never inside text fields.
The active task's shortcut strip and help drawer show the current meanings.

- **Paint:** B/E/I brush; brackets resize; 1 Network, 2 Classical, 3 Tube fit,
  4 raw Threshold, 5 Saved; H/L/D/S refinement; Z draft undo; N next, P visited
  previous; Space/Enter Save + next; F raw/flat; O opacity cycle; 0 fit view.
  Ctrl/Cmd+Z undoes the draft in Paint, never silently a saved workspace edit.
- **Other tasks:** Space playback; brackets flagged navigation; 1–9 layers;
  N notes; S fitter starts; existing arrow-key frame stepping and modifiers.
- Undo saved operations is explicitly labeled in History. Proposal modifiers
  apply to proposal buttons/keys; Shift-drag on the canvas pans, and Shift-drag
  on the timeline selects a range. Right/middle/Alt-drag canvas pan remains safe.

## Implementation sequence and acceptance

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
