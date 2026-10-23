"""Meaningful synthetic correctness tests, NOT actual SWEET.TV results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import unittest
import warnings

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from sweettv.models import PopularityModel, ContentModel, ImplicitALSModel, CoViewPPRModel


def metadata(n):
    return [{"item_id": i + 1, "title": "", "category": "", "genre_1": "", "genre_2": "",
             "genre_3": "", "year": "", "director": "", "actors": ""} for i in range(n)]


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.X = sparse.csr_matrix(np.array([[3, 1, 0, 0, 0], [1, 2, 0, 0, 0],
            [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
        self.meta = metadata(5)
        self.meta[0].update(title="Cats", category="Animals", genre_1="Family")
        self.meta[1].update(title="Dogs", category="Animals", genre_1="Family")
        self.meta[2].update(title="Robots", category="Science", genre_1="Technology")
        self.meta[3].update(title="Local bulletin", category="News")
        self.meta[4].update(title="Future quasar secretword", category="Animals", genre_1="Family")
        self.users = np.arange(6, dtype=np.int64)
        self.items = np.arange(5, dtype=np.int64)

    def test_popularity_counts_distinct_users_not_repeated_duration(self):
        model = PopularityModel().fit(self.X, self.meta)
        np.testing.assert_array_equal(model.item_scores_, [2, 3, 2, 1, 0])
        np.testing.assert_array_equal(model.score(np.array([0, 5, -1, 999]), self.items),
            [[2, 3, 2, 1, 0], [0]*5, [0]*5, [0]*5])
        weighted = PopularityModel(mode="weight").fit(self.X, self.meta)
        np.testing.assert_array_equal(weighted.item_scores_, [4, 4, 2, 2, 0])

    def test_content_future_words_do_not_fit_vocabulary_or_idf(self):
        model = ContentModel().fit(self.X, self.meta)
        vocab = dict(model.vectorizer_.vocabulary_)
        idf = model.vectorizer_.idf_.copy()
        self.assertNotIn("title:secretword", vocab)
        changed = [dict(row) for row in self.meta]
        changed[-1].update(title="completely new vocab futureonly leaky", category="Newcategory", genre_1="Newgenre")
        other = ContentModel().fit(self.X, changed)
        self.assertEqual(vocab, other.vectorizer_.vocabulary_)
        np.testing.assert_array_equal(idf, other.vectorizer_.idf_)
        # An unseen programme can still share known TRAIN attributes.
        self.assertGreater(model.score(np.array([0]), np.array([4]))[0, 0], 0)
        self.assertEqual(other.score(np.array([0]), np.array([4]))[0, 0], 0)

    def test_content_cosine_matches_hand_computed_attribute_case(self):
        # Two one-feature programmes with equal IDF; the TRAIN profile is [3,4].
        X = sparse.csr_matrix(np.array([[3, 4, 0], [0, 0, 0]], dtype=np.float32))
        meta = metadata(3)
        meta[0]["category"] = "A"
        meta[1]["category"] = "B"
        meta[2]["category"] = "A"  # Future item shares a known feature.
        model = ContentModel(interaction_transform="identity", title_weight=0,
            category_weight=1, genre_weight=0, people_weight=0, year_weight=0).fit(X, meta)
        actual = model.score(np.array([0, 1]), np.array([0, 1, 2]))
        np.testing.assert_allclose(actual, [[.6, .8, .6], [0, 0, 0]], atol=1e-7)

    def test_all_models_preserve_input_and_candidate_order(self):
        original = (self.X.data.copy(), self.X.indices.copy(), self.X.indptr.copy())
        old_meta = json.dumps(self.meta, sort_keys=True)
        for model in [PopularityModel(), ContentModel(), CoViewPPRModel(alpha=.5),
                      ImplicitALSModel(factors=3, iterations=5, num_threads=1, seed=17)]:
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.meta)
                all_scores = model.score(np.array([0, 5]), self.items)
                subset = model.score(np.array([0, 5]), np.array([4, 1, 4, 0]))
                self.assertEqual(subset.shape, (2, 4))
                self.assertTrue(np.isfinite(all_scores).all())
                np.testing.assert_allclose(subset, all_scores[:, [4, 1, 4, 0]], atol=1e-7)
                np.testing.assert_array_equal(all_scores[1], np.zeros(5))
                self.assertEqual(model.score(np.array([], dtype=int), self.items).shape, (0, 5))
                self.assertEqual(model.score(np.array([0]), np.array([], dtype=int)).shape, (1, 0))
                np.testing.assert_array_equal(self.X.data, original[0])
                np.testing.assert_array_equal(self.X.indices, original[1])
                np.testing.assert_array_equal(self.X.indptr, original[2])
                self.assertEqual(old_meta, json.dumps(self.meta, sort_keys=True))

    def test_als_fits_users_by_items_and_preserves_repeat_eligibility(self):
        # Two disjoint co-view groups; the first user's previously watched item
        # should stay eligible and the unseen item of the other group score lower.
        X = sparse.csr_matrix(np.array([[4, 3, 0, 0, 0], [3, 4, 0, 0, 0],
            [0, 0, 4, 3, 0], [0, 0, 3, 4, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
        model = ImplicitALSModel(factors=2, iterations=30, alpha=10,
            regularization=.1, seed=123, num_threads=1, use_cg=False).fit(X, metadata(5))
        scores = model.score(np.arange(5), np.arange(5))
        self.assertEqual(model.model_.user_factors.shape, (5, 2))
        self.assertEqual(model.model_.item_factors.shape, (5, 2))
        self.assertGreater(scores[0, 0], scores[0, 2])
        self.assertGreater(scores[2, 2], scores[2, 0])
        self.assertGreater(scores[0, 0], 0)  # No seen-item mask.
        np.testing.assert_array_equal(scores[4], 0)
        np.testing.assert_array_equal(scores[:, 4], 0)
        self.assertAlmostEqual(model.diagnostics_["confidence_min"], 1 + 10 * np.log1p(3), places=5)

    def test_als_fixed_seed_single_thread_is_reproducible(self):
        args = dict(factors=3, iterations=8, alpha=12, regularization=.2, seed=17, num_threads=1)
        a = ImplicitALSModel(**args).fit(self.X, self.meta)
        b = ImplicitALSModel(**args).fit(self.X, self.meta)
        np.testing.assert_array_equal(a.score(self.users, self.items), b.score(self.users, self.items))

    def test_ppr_matches_independent_dense_linear_system_and_conserves_mass(self):
        model = CoViewPPRModel(alpha=.7, tol=1e-12, max_iter=500).fit(self.X, self.meta)
        result = model.score(self.users, self.items)
        Q = model.transition_.toarray()
        np.testing.assert_allclose(Q.sum(axis=1), [1, 1, 1, 0, 0], atol=1e-12)
        for user in range(5):
            v = model.restart_[user].toarray().ravel()
            # Independent non-iterative oracle: explicitly materialize dangling
            # rows as v, then solve the stationary linear system.
            transition = Q.copy()
            transition[model.dangling_] = v
            oracle = np.linalg.solve(np.eye(5) - .7 * transition.T, .3 * v)
            np.testing.assert_allclose(result[user], oracle, atol=5e-8)
            self.assertAlmostEqual(float(result[user].sum()), 1, places=6)
        self.assertTrue(all(x["converged"] for x in model.last_convergence_))
        self.assertEqual(model.diagnostics_["dangling_items"], 2)
        # Restarting at a dangling item is one-hot stationary, NOT global/uniform.
        np.testing.assert_array_equal(result[4], [0, 0, 0, 1, 0])

    def test_ppr_alpha_zero_and_iteration_cap_are_reported_honestly(self):
        zero = CoViewPPRModel(alpha=0, tol=1e-10).fit(self.X, self.meta)
        scores = zero.score(np.array([0]), self.items)
        np.testing.assert_allclose(scores[0], [2/3, 1/3, 0, 0, 0], atol=1e-7)
        self.assertEqual(zero.last_convergence_[0]["iterations"], 1)
        capped = CoViewPPRModel(alpha=.95, max_iter=1, tol=1e-12).fit(self.X, self.meta)
        capped.score(np.array([0]), self.items)
        self.assertFalse(capped.last_convergence_[0]["converged"])
        self.assertEqual(capped.last_convergence_[0]["iterations"], 1)
        self.assertGreater(capped.last_convergence_[0]["residual"], capped.tol)

    def test_ppr_block_and_score_batch_sizes_do_not_change_results(self):
        a = CoViewPPRModel(alpha=.7, graph_block_size=1, score_batch_size=1).fit(self.X, self.meta)
        b = CoViewPPRModel(alpha=.7, graph_block_size=3, score_batch_size=4).fit(self.X, self.meta)
        np.testing.assert_allclose(a.transition_.toarray(), b.transition_.toarray(), atol=1e-12)
        np.testing.assert_array_equal(a.score(self.users, self.items), b.score(self.users, self.items))

    def test_ppr_pruning_ties_are_broken_by_lower_item_index(self):
        X = sparse.csr_matrix(np.ones((1, 4), dtype=np.float32))
        model = CoViewPPRModel(neighbors=1).fit(X, metadata(4))
        self.assertEqual(model.transition_.nnz, 4)
        np.testing.assert_array_equal(model.transition_.toarray(), [[0,1,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]])

    def test_empty_training_and_missing_content_are_explicit_zeros(self):
        empty = sparse.csr_matrix((3, 4), dtype=np.float32)
        for model in [PopularityModel(), ContentModel(), ImplicitALSModel(factors=2, iterations=2), CoViewPPRModel()]:
            model.fit(empty, metadata(4))
            np.testing.assert_array_equal(model.score(np.arange(3), np.arange(4)), np.zeros((3,4)))

    def test_invalid_weights_and_indices_fail_instead_of_silent_nan(self):
        for value in [-1., np.nan, np.inf]:
            X = sparse.csr_matrix([[value]], dtype=np.float64)
            with self.assertRaises(ValueError):
                ContentModel().fit(X, metadata(1))
        model = ContentModel().fit(self.X, self.meta)
        with self.assertRaises(IndexError):
            model.score(np.array([0]), np.array([5]))
        with self.assertRaises(TypeError):
            model.score(np.array([.5]), self.items)
        with self.assertRaises(RuntimeError):
            ContentModel().score(np.array([0]), np.array([0]))


if __name__ == "__main__":
    import scipy, sklearn, implicit
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"scope": "AI-run synthetic correctness tests; not a student personal check or SWEET.TV result",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "successful": result.wasSuccessful(), "seconds": time.perf_counter()-start,
        "versions": {"numpy":np.__version__, "scipy":scipy.__version__, "sklearn":sklearn.__version__, "implicit":implicit.__version__},
        "models_sha256": hashlib.sha256((Path(__file__).resolve().parents[1]/"src/sweettv/models.py").read_bytes()).hexdigest(),
        "failure_details": [(str(t),msg) for t,msg in result.failures+result.errors]}
    Path(__file__).with_name("synthetic_test_results.json").write_text(json.dumps(payload, indent=2))
    raise SystemExit(not result.wasSuccessful())
