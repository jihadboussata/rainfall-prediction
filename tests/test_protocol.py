import unittest
import numpy as np
import pandas as pd
from rainfall.data import date_folds, features, partitions
from rainfall.train import choose_threshold, preprocessing


class ProtocolTests(unittest.TestCase):
    def test_dates_are_disjoint_and_strictly_ordered(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=100).repeat(3))
        parts = partitions(dates)
        self.assertEqual(sum(map(len, parts)), len(dates))
        for left, right in zip(parts[:-1], parts[1:]):
            self.assertLess(dates.iloc[left].max(), dates.iloc[right].min())
        for train, valid in date_folds(dates.iloc[parts[0]]):
            self.assertLess(dates.iloc[train].max() + pd.Timedelta(days=1), dates.iloc[valid].min())

    def test_preprocessing_learns_only_the_fit_rows(self):
        train = pd.DataFrame({"Temperature": [1., 3., np.nan], "Location": ["A", "B", "A"]})
        transformer = preprocessing(train).fit(train)
        before = transformer.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
        transformed = transformer.transform(pd.DataFrame({"Temperature": [1000.], "Location": ["Unseen"]}))
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(before, [2.])
        np.testing.assert_array_equal(before, transformer.named_transformers_["numeric"].named_steps["impute"].statistics_)

    def test_target_is_excluded_and_today_is_preserved(self):
        frame = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]), "RainToday": ["Yes"], "RainTomorrow": ["No"]})
        self.assertNotIn("RainTomorrow", features(frame))
        self.assertNotIn("Date", features(frame))
        self.assertEqual(features(frame).RainToday.iloc[0], "Yes")

    def test_validation_threshold_has_expected_decision(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .2, .6, .8])
        threshold = choose_threshold(y, p)
        np.testing.assert_array_equal(p >= threshold, y)


if __name__ == "__main__":
    unittest.main()
