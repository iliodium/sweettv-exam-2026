"""Synthetic unit/property checks; no final competition CSV is created here."""
import csv
import itertools
import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from sweettv.metrics import (
    average_precision_at_k, evaluate_rankings, ranking_metrics_at_k,
    stable_topk, validate_ranked_lists, validate_submission_csv,
)


def rational_ap_oracle(relevant, predicted, k):
    actual = frozenset(relevant)
    if not actual:
        return None
    numerator = Fraction(0)
    for position in range(min(k, len(predicted))):
        item = predicted[position]
        if item in actual and predicted.index(item) == position:
            distinct_prefix_hits = len(actual.intersection(predicted[:position+1]))
            numerator += Fraction(distinct_prefix_hits, position+1)
    return numerator / min(k, len(actual))


class MetricTests(unittest.TestCase):
    def test_lecture_style_fixtures(self):
        self.assertAlmostEqual(average_precision_at_k([1,2,3],[1,2,9,3,8]), 11/12)
        self.assertAlmostEqual(average_precision_at_k([1,2,3],[9,1,8,2,3]), 8/15)

    def test_denominator_cap_and_precision_recall_distinction(self):
        value = ranking_metrics_at_k(range(1,9), [1,2,3,4,5])
        self.assertEqual(value["ap"], 1)
        self.assertEqual(value["precision"], 1)
        self.assertEqual(value["recall"], 5/8)
        self.assertEqual(average_precision_at_k(range(1,9), [1,2,3,4,99]), .8)

    def test_duplicate_prediction_positions_get_no_repeated_credit(self):
        self.assertAlmostEqual(average_precision_at_k([1,2],[1,1,2,9,9]), 5/6)
        self.assertEqual(average_precision_at_k([1],[1,1,1,1,1]), 1)
        self.assertEqual(average_precision_at_k([1,1,1],[1,2,3,4,5]), 1)
        self.assertEqual(ranking_metrics_at_k([1,2],[1,1,2,9,9])["hits"],2)

    def test_no_relevant_users_excluded_but_not_silently_missing_predictions(self):
        result=evaluate_rankings({"u1":[1,2,3],"empty":[],"u2":[1,2,3]},
          {"u1":[1,2,9,3,8],"u2":[9,1,8,2,3]}, ["empty","u1","u2","absent"])
        self.assertEqual(result["considered_users"],4)
        self.assertEqual(result["evaluated_users"],2)
        self.assertEqual(result["excluded_no_relevant_users"],["empty","absent"])
        self.assertAlmostEqual(result["map"],29/40)
        with self.assertRaisesRegex(ValueError,"Missing predictions"):
            evaluate_rankings({"u1":[1]}, {})
        self.assertIsNone(evaluate_rankings({"u1":[]},{})["map"])

    def test_only_first_k_ranks_and_short_list_penalty(self):
        self.assertEqual(average_precision_at_k([8],[1,2,3,4,5,8]),0)
        value=ranking_metrics_at_k([1,2,3],[1])
        self.assertEqual(value["ap"],1/3)
        self.assertEqual(value["precision"],1/5)
        self.assertEqual(value["recall"],1/3)

    def test_exhaustive_sequences_against_exact_fraction_oracle(self):
        truth_sets=[s for count in range(1,5) for s in itertools.combinations(range(1,5),count)]
        compared=0
        for predicted in itertools.product(range(1,5), repeat=5):
            for relevant in truth_sets:
                expected=rational_ap_oracle(relevant,list(predicted),5)
                self.assertAlmostEqual(average_precision_at_k(relevant,predicted),float(expected),places=14)
                compared+=1
        self.assertEqual(compared,15360)

    def test_stable_topk_uses_raw_id_ties_not_input_order(self):
        self.assertEqual(stable_topk(np.array([.2,.8,.8,.8]),np.array([91,30,10,20]),3),[10,20,30])
        self.assertEqual(stable_topk([0,-0.0,0],[9,1,4],5),[1,4,9])
        self.assertEqual(stable_topk([],[],5),[])

    def test_stable_topk_invalid_inputs_fail(self):
        for scores,ids in [([1],[0]),([1],[-1]),([1],[1.0]),([1],[True]),([1,2],[1,1]),([1],[1,2]),([math.nan],[1]),([math.inf],[1]),([-math.inf],[1]),([True],[1])]:
            with self.subTest(scores=scores,ids=ids),self.assertRaises((TypeError,ValueError)):
                stable_topk(scores,ids)

    def test_metric_invalid_ids_and_k_fail(self):
        for value in [0,-1,1.0,math.nan,True,"1"]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                average_precision_at_k([value],[1,2,3,4,5])
        for k in [0,-1,5.0,True]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                average_precision_at_k([1],[1],k)

    def test_valid_ranked_lists_preserve_unsigned_64_bit_user_strings(self):
        users=["18417531283109304442","8986460610792053980"]
        result=validate_ranked_lists(users,{users[0]:[1,2,3,4,5],users[1]:[5,4,3,2,1]},range(1,7))
        self.assertEqual(result["first_user"],users[0]);self.assertEqual(result["ranked_items"],10)

    def test_bad_ranked_lists_fail_with_useful_errors(self):
        cases=[({"a":[1,2,3,4]},"exactly"),({"a":[1,1,3,4,5]},"repeated"),({"a":[1,2,3,4,9]},"outside"),({},"missing"),({"a":[1,2,3,4,5],"extra":[1,2,3,4,5]},"extra"),({"a":[0,2,3,4,5]},"unlabelled")]
        for predictions,message in cases:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                validate_ranked_lists(["a"],predictions,range(1,7))
        with self.assertRaisesRegex(ValueError,"duplicates"):
            validate_ranked_lists(["a","a"],{"a":[1,2,3,4,5]},range(1,7))

    def test_existing_csv_exact_header_order_and_hash(self):
        # Synthetic temporary fixtures are deliberately not competition outputs.
        with tempfile.TemporaryDirectory(prefix="synthetic_metric_test_") as temp:
            sample=Path(temp)/"sample.csv";output=Path(temp)/"candidate.csv"
            users=["18417531283109304442","8986460610792053980"]
            def write(path,header,rows):
                with path.open("w",newline="") as handle:
                    writer=csv.writer(handle);writer.writerow(header);writer.writerows(rows)
            write(sample,["user_id","tv_show_id"],[(u,"0 0 0 0 0") for u in users])
            write(output,["user_id","tv_show_id"],[(u,"1 2 3 4 5") for u in users])
            report=validate_submission_csv(output,sample,range(1,7))
            self.assertEqual(report["users"],2);self.assertTrue(report["row_order_matches_sample"]);self.assertEqual(len(report["sha256"]),64)
            write(output,["tv_show_id","user_id"],[("1 2 3 4 5",u) for u in users])
            with self.assertRaisesRegex(ValueError,"header/order"):
                validate_submission_csv(output,sample,range(1,7))
            write(output,["user_id","tv_show_id"],[(u,"1 2 3 4 5") for u in reversed(users)])
            with self.assertRaisesRegex(ValueError,"row order"):
                validate_submission_csv(output,sample,range(1,7))


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"scope":"Independent synthetic mathematical/validation tests; no competition submission generated", "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"all_pass":result.wasSuccessful(),"rational_oracle_sequences":15360,"fixture_ap":[11/12,8/15],"map_fixture":29/40}
    Path(__file__).with_name("metric_tests.json").write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
