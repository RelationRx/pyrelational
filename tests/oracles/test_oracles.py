from unittest import TestCase

from pyrelational.oracles.benchmark_oracle import BenchmarkOracle
from tests.test_utils import get_classification_dataset


class TestOracle(TestCase):
    """Class containing unit tests for oracles"""

    def setUp(self) -> None:
        """Set up datamager."""
        self.datamanager = get_classification_dataset()

    def test_update_annotations(self) -> None:
        """Check update_annotations method updates unlabelled and labelled sets."""
        random_u_sindex = self.datamanager.u_indices[0]
        len_dm_l = len(self.datamanager.l_indices)
        len_dm_u = len(self.datamanager.u_indices)

        BenchmarkOracle.update_annotations(self.datamanager, [random_u_sindex])
        self.assertIn(random_u_sindex, self.datamanager.l_indices)
        self.assertGreater(len(self.datamanager.l_indices), len_dm_l)
        self.assertGreater(len_dm_u, len(self.datamanager.u_indices))

    def test_query_target_value(self) -> None:
        """Check query target value of benchmark oracle return correct values."""
        oracle = BenchmarkOracle()
        value = oracle.query_target_value(self.datamanager, 0)
        self.assertEqual(value, self.datamanager[0][-1])

    def test_update_target_value(self) -> None:
        """Check update_target_value method updates dataset correctly."""
        BenchmarkOracle.update_target_value(self.datamanager, 0, 42)
        self.assertEqual(self.datamanager[0][-1], 42)

    def test_update_target_values(self) -> None:
        """Test that update_target_values method change all values in dataset."""
        ixs, vals = [0, 1, 2], [42, 42, 42]
        BenchmarkOracle.update_target_values(self.datamanager, ixs, vals)
        self.assertEqual([self.datamanager[i][-1] for i in ixs], vals)
