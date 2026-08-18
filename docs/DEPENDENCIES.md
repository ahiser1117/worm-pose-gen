# Dependencies

The runtime dependency set is intentionally limited to NumPy, h5py, PyTorch,
PyTorch Lightning, and Matplotlib.

- **NumPy** provides array preprocessing and deterministic numeric utilities.
- **h5py** is required for bounded source reads and streamed output HDF5.
- **PyTorch** provides the geometry, model, refinement, and inference runtime.
- **PyTorch Lightning** provides the reusable training module and controlled
  training/checkpoint integration required by the project specification.
- **Matplotlib** produces auditable experiment and final figures with the
  non-interactive `Agg` backend.

No tracking service, cloud dependency, general image-processing package, or
notebook runtime is part of the environment. A research-only dependency may be
added only after a recorded experiment shows that implementing the equivalent
locally carries more scientific or engineering risk.
