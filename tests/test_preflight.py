from __future__ import annotations

import unittest

from scripts.preflight import parse_nvidia_smi_row


class NvidiaSmiParsingTest(unittest.TestCase):
    def test_parse_expected_single_device_row(self) -> None:
        parsed = parse_nvidia_smi_row(
            "0, GPU-f72d2ba7-8334-183e-e368-2c527e8a39e6, "
            "00000000:01:00.0, NVIDIA RTX 6000 Ada Generation, "
            "49140 MiB, 610.43.02\n"
        )

        self.assertEqual(parsed["physical_index"], 0)
        self.assertEqual(
            parsed["uuid"], "GPU-f72d2ba7-8334-183e-e368-2c527e8a39e6"
        )
        self.assertEqual(parsed["pci_bus_id"], "00000000:01:00.0")
        self.assertEqual(parsed["name"], "NVIDIA RTX 6000 Ada Generation")
        self.assertEqual(parsed["total_memory"], "49140 MiB")
        self.assertEqual(parsed["driver_version"], "610.43.02")

    def test_reject_multiple_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one nvidia-smi row"):
            parse_nvidia_smi_row(
                "0, uuid0, pci0, gpu0, 1 MiB, driver\n"
                "1, uuid1, pci1, gpu1, 1 MiB, driver\n"
            )

    def test_reject_non_numeric_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid physical GPU index"):
            parse_nvidia_smi_row("zero, uuid, pci, gpu, 1 MiB, driver\n")


if __name__ == "__main__":
    unittest.main()
