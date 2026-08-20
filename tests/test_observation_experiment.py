import unittest

from scripts.exp_smc_005_observation_energy import _curve_gate_summary


class ObservationExperimentAggregationTests(unittest.TestCase):
    def test_symmetric_basin_counts_all_six_outward_steps(self) -> None:
        perturbations = ["translation_x", "rotation", "shape", "length"]
        cases = [
            {
                "perturbations": {
                    name: {"energies": {"energy": [3, 2, 1, 0, 1, 2, 3]}}
                    for name in perturbations
                }
            }
        ]
        result = _curve_gate_summary(cases, "energy", perturbations)
        self.assertEqual(result["overall_near_zero_minimum_fraction"], 1.0)
        self.assertEqual(result["overall_outward_monotonic_step_fraction"], 1.0)
        for values in result["endpoint_minus_zero_energy_by_perturbation"].values():
            self.assertEqual(values["median"], 3.0)

    def test_argmin_tie_uses_first_index(self) -> None:
        cases = [
            {
                "perturbations": {
                    "translation_x": {"energies": {"energy": [1, 1, 1, 1, 1, 1, 1]}}
                }
            }
        ]
        result = _curve_gate_summary(cases, "energy", ["translation_x"])
        self.assertEqual(result["overall_near_zero_minimum_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
