# Worm pose

The pose app finds the body of a *C. elegans* in every frame of a behaviour
recording (NIR video): its midline from head to tail, its width, and which
end is the head. It flags the stretches it is unsure of so you can check and
fix them, and exports one table per recording with the midline, curvature,
width, head and tail, centroid and velocity of every frame. You can also
label frames and train your own models on the lab's labels plus yours.

It runs on the flv-c machines and on Engaging. The lab's models and labels
come from a shared, read-only lab library; everything you make (labels,
models, analyses) stays in your own data directory.

## Install

You need read access to this GitHub repository and
[`uv`](https://docs.astral.sh/uv/). If `uv --version` fails, install it with
`curl -LsSf https://astral.sh/uv/install.sh | sh` and open a new shell.

```bash
git clone git@github.com:ahiser1117/worm-pose-gen.git
cd worm-pose-gen
scripts/bootstrap_environment.sh --data-dir /temp_data4/$USER/worm-pose-data
```

The checkout can go anywhere; with its Python environment it takes about
5 GB. `--data-dir` is where your work will be saved. An analysed recording
takes about 2 GB, so choose a folder on large storage: on the flv machines
your folder in `/temp_data4` if you have one, on Engaging a folder in the
lab's project storage under `/orcd/data/flavell/001/` rather than your home
directory. Without `--data-dir` it is `user_data/` inside the checkout.

Bootstrap installs Python and the dependencies into the checkout (a couple of
minutes), saves the data directory in `settings.json`, and finishes with a
check of the install that should end in `OK`, like this (shortened):

```
ok    Python 3.13.15 (3.13 required)
ok    lab library /storage/fs/store1/shared/worm-pose-models
ok    lab:nir-flv default lab:arch-combo-self-fib
ok    data directory /temp_data4/you/worm-pose-data (from .../settings.json)
ok    612 GB free for the data directory (an analysed recording takes about 2 GB)
ok    GPUs: 0: NVIDIA RTX A6000, ...
OK
```

A `warn` line is something the app works without (no GPU on this machine,
little free space); a `FAIL` line has to be fixed first.

## Start the app

On an flv machine, from the checkout:

```bash
./worm-pose-app
```

It prints where it is and how to open it from your own computer:

```
pose app at http://127.0.0.1:8768/
From your own computer: ssh -N -L 8768:localhost:8768 you@flv-c3
then open http://localhost:8768/
```

Run that `ssh` command in a terminal on your computer, leave it open, and
open the address in your browser. The app listens only on the machine it
runs on and has no login, so it is reachable through such a tunnel, or by
anyone logged in to that same machine. If someone else's app has port 8768
the next free port is taken, and the printed commands say which. Stop the
app with Ctrl-C.

On **Engaging**, start the app inside an interactive GPU job rather than on
the login node:

```bash
srun -p ou_bcs_normal --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 --pty bash
cd worm-pose-gen
./worm-pose-app
```

The printed `ssh` command goes through the login node to your job's node.
From the app you can also send analyses and training to SLURM (**Run on**),
so they keep running after your interactive job ends.

The app uses the GPUs of the machine it runs on, choosing the least busy one
for each analysis; `./worm-pose-app --gpus 2,3` limits it to some.
`./worm-pose-app --help` lists the other options.

## First steps

1. **Add your recordings.** The first screen asks for the folder your
   recordings are in. Every `.h5` recording and `.avi` video in it, and in
   the folders up to two levels below it, is listed: a folder of days, each
   holding recordings, works. The app checks each new file in the
   background; a file that cannot be read says so. **Add recordings…** adds
   more folders or single files later, or removes a folder.
2. **Open a recording.** This makes its workspace. A video is first
   converted to an HDF5 recording (about a minute per ten minutes of video).
3. **Analyse.** The form on the left uses the lab's default models; press
   **Analyse**. A progress bar per stage follows the analysis, which takes
   roughly a minute per thousand frames on one GPU. You can close the browser
   and come back.
4. **Review the issues.** The list on the left holds the stretches the app is
   unsure of, each with a reason ("head/tail uncertain", "coiled", "mask fits
   poorly", "leaves the view"). Mark one **Looks OK** (O) or fix it, and go
   to the next. Press `?` for the keyboard shortcuts.
5. **Export.** **Export** in the header writes the table of the recording
   (see [Exports](#exports)).

## The Workspace page

The **Recordings** screen lists the recordings of a setup (a microscope; the
lab's NIR tracking rigs are `NIR behaviour camera (flv rigs)`): first the
ones you have analysed, most recently opened first, then the rest. Each
shows its length and status: not analysed, analysing, *N* issues to review,
reviewed or exported.

A recording's workspace shows the frame with four layers (keys 1–4: mask,
midline with the head a square and the tail a circle, outline, and the A-P
field from head to tail), the issues on the left, and at the bottom the
transport, the issue track and the curvature kymograph. Drag on the
kymograph to scrub; right-drag to select a range for a fix. The fixes are:

- **Flip** swaps head and tail, for the whole issue or one frame.
- **Refit** fits the stretch again with a method chosen from the issue's
  reasons, and shows before and after so you keep or discard it.
- **Edit mask** paints the worm on one frame; saving refits around it.
- **Relabel** sends keyframes of the stretch to Labeling; once they are
  labeled the stretch is rebuilt from them.

Every fix is listed under the fixes, with Undo. **Re-analyse…** in the
header runs the analysis again, for example with another model.

### Exports

**Export** writes `exports/<recording>_<time>/` in the recording's workspace
(in your data directory): a Parquet table with one row per frame, and
`export.json` with the units, the models and the setup. Columns: frame,
time, the frame's status (automatic, reviewed, fixed or unresolved), the
midline from head to tail, its curvature, the width profile, head and tail
positions, centroid and velocity. Lengths are in µm and positions in image
coordinates (origin at the top left, y down). Each column's unit and
description are also stored in the Parquet file itself.

## The Labeling page

Labels teach the models. A label is one frame's mask (which pixels are
worm) and its body (the midline from head to tail). Draw the mask with the
Worm and Background brushes, or start from the model's proposal; then trace
the midline from the head, or use the proposal; **Save & next** (Enter).
Frames come from queues: a workspace's Relabel keyframes, **New queue** (a
spread of frames over chosen recordings, favouring the ones the model is
least sure of), or **Browse labels**. Your labels are saved in your data
directory; the lab's labels stay as they are, and editing one saves your own
newer version.

## The Training page

**Models** lists the lab's models and yours with their scores on a
benchmark (mask IoU, head/tail correct, A-P error); ★ marks the setup's
defaults. **Use as default** makes a model your default for a setup (only
for you). **Train from this** starts a training run from a model.
**Datasets** shows the labels by recording and split: a dataset says which
recordings train, validate or test a model, so it can combine the lab's
labels with yours. A trained model is evaluated on the setup's benchmarks
and appears in Models.

## Where things are

| | |
|---|---|
| Your data directory (`settings.json`) | `library/` your labels, datasets, models and recording folders; `workspaces/` one per opened recording, with its exports; `cache/` flat fields and body-size priors, rebuilt when missing; `runs/` outputs of command-line scripts |
| Lab library (read-only) | `/storage/fs/store1/shared/worm-pose-models` on flv, `/orcd/data/flavell/001/shared/worm-pose-models` on Engaging |
| The checkout | the code, `.venv` (Python environment), `settings.json` |

To change the data directory, move its contents if you want to keep them,
run `scripts/bootstrap_environment.sh --data-dir <new folder>` and start the
app again.

## Updating

```bash
git pull
scripts/bootstrap_environment.sh
```

Bootstrap keeps the data directory named in `settings.json`.

## Troubleshooting

- **A recording shows "Cannot be read".** Hover over it for the reason. Some
  older recordings are damaged on disk, and some need an HDF5 compression
  plugin that is not installed.
- **The check says the lab library is not readable.** The lab library is
  known on the flv-c machines and Engaging only; elsewhere you have only your
  own models.
- **Analyse or Train says there is no GPU.** The machine has none the app can
  use: start it on a machine with a GPU, or choose SLURM under Run on.
  Labeling works without a GPU, but the model's proposal then takes 15–20 s
  per frame and is only made when you ask (**Propose**).
- **The browser shows nothing.** Check that the `ssh` command is still
  running and that the port in the address is the one the app printed.

## For developers

[`docs/DEVELOPER.md`](docs/DEVELOPER.md) covers the tests, training and
evaluating from the command line, publishing models and labels to the lab
library, the HTTP API, developer mode (`--dev`), and the research history of
the pipeline. The app's design is
[`docs/APP_SIMPLIFICATION.md`](docs/APP_SIMPLIFICATION.md).
