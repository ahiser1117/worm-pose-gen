import unittest

from scripts.build_exp_003_proxy_exclusions import merge_intervals


class Exp003ProxyExclusionTests(unittest.TestCase):
    def test_merge_intervals(self):
        self.assertEqual(
            merge_intervals({1, 2, 3, 8, 10, 11}),
            [[1, 3], [8, 8], [10, 11]],
        )

    def test_empty_intervals(self):
        self.assertEqual(merge_intervals(set()), [])


if __name__ == "__main__":
    unittest.main()
