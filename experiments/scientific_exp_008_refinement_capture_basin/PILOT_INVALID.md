# Invalid untuned pilot

This directory preserves the first mechanics sweep for auditability. Its Adam
trust radii were larger than several declared initialization perturbations, so
the first update displaced already-good poses and the sweep does not measure a
valid capture basin. It was diagnosed before interpretation and superseded by
`../scientific_exp_008_refinement_capture_basin_tuned_n8/`.
