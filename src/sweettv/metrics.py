"""Deterministic ranking metrics and output checks for the SWEET.TV exam.

AP@K uses the Kaggle-style denominator min(K, number of DISTINCT relevant
programme IDs). Duplicate predictions consume rank positions but receive no
repeated credit. A user with no relevant programmes has undefined AP/recall:
``evaluate_rankings`` excludes that user and reports the exclusion explicitly.

Programme ID 0 is unlabelled airtime, not an item. All programme IDs accepted
here are positive integers. These functions never fill a short recommendation
list, select a model, or invent ground truth; a common fallback belongs in the
caller and must be applied identically to every compared method.
"""
from __future__ import annotations

import math
import csv
import hashlib
import re
from pathlib import Path
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Hashable, Any


def _valid_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return int(k)


def _programme_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer programme ID; 0 is unlabelled airtime, got {value!r}")
    return int(value)


def _programme_ids(values: Iterable[int], *, label: str) -> list[int]:
    if isinstance(values, (str, bytes)) or values is None:
        raise TypeError(f"{label} must be an iterable of programme IDs, not a string/None")
    try:
        return [_programme_id(value, label=f"{label}[{position}]") for position, value in enumerate(values)]
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable of programme IDs") from error


def _user_id(value: Any, *, label: str) -> Hashable:
    # Preserve raw integer/string identity. In particular, never strip leading
    # zeroes from a string ID or silently coerce a float/NaN to an integer.
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer or nonempty string, got bool")
    if isinstance(value, Integral) and value >= 0:
        return int(value)
    if isinstance(value, str) and value and value.strip() == value:
        return value
    raise ValueError(f"{label} must be a nonnegative integer or nonempty string, got {value!r}")


def _user_ids(values: Iterable[Hashable]) -> list[Hashable]:
    if isinstance(values, (str, bytes)):
        raise TypeError("user_ids must be an iterable of IDs, not a single string")
    out = [_user_id(value, label=f"user_ids[{i}]") for i, value in enumerate(values)]
    if len(set(out)) != len(out):
        raise ValueError("user_ids contains duplicates")
    return out


def average_precision_at_k(relevant: Iterable[int], predicted: Iterable[int], k: int = 5) -> float | None:
    """Return AP@K, or None for a user with no distinct relevant programmes.

    All supplied IDs are validated, while only the first K prediction positions
    contribute to AP. Duplicate ground truth is collapsed. Duplicate predicted
    IDs are allowed for metric diagnostics and cannot receive repeated credit;
    ``validate_ranked_lists`` separately rejects them in a real output file.
    """
    k = _valid_k(k)
    actual = set(_programme_ids(relevant, label="relevant"))
    prediction = _programme_ids(predicted, label="predicted")
    if not actual:
        return None
    seen: set[int] = set()
    hits = 0
    precision_sum = 0.0
    for rank, programme_id in enumerate(prediction[:k], start=1):
        if programme_id in actual and programme_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(programme_id)
    return precision_sum / min(k, len(actual))


def ranking_metrics_at_k(relevant: Iterable[int], predicted: Iterable[int], k: int = 5) -> dict[str, Any]:
    """Return AP, precision, recall and counts for one user.

    Precision has denominator K, even for an invalid short list. Recall has
    denominator the entire distinct relevant set (not min(K, |relevant|)).
    Empty-ground-truth users have None for all three metrics and are excluded
    from their macro averages by ``evaluate_rankings``.
    """
    k = _valid_k(k)
    actual = set(_programme_ids(relevant, label="relevant"))
    prediction = _programme_ids(predicted, label="predicted")
    hits = len(actual.intersection(prediction[:k]))
    defined = bool(actual)
    return {
        "ap": average_precision_at_k(actual, prediction, k),
        "precision": hits / k if defined else None,
        "recall": hits / len(actual) if defined else None,
        "hits": hits,
        "relevant_count": len(actual),
        "prediction_count": len(prediction),
        "evaluated_positions": min(k, len(prediction)),
        "duplicate_predictions_in_topk": len(prediction[:k]) - len(set(prediction[:k])),
    }


def evaluate_rankings(
    ground_truth: Mapping[Hashable, Iterable[int]],
    predictions: Mapping[Hashable, Iterable[int]],
    user_ids: Iterable[Hashable] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Macro-average metrics on an explicit common cohort and report exclusions.

    With ``user_ids=None``, consider every ground-truth mapping key. An explicit
    cohort permits predictions for a larger population; extra prediction keys
    are ignored here. Missing predictions for a user with relevant ground truth
    raise an error, rather than silently dropping a hard-to-recommend user.
    Users absent from ground_truth are treated as having no relevant programme.
    No eligible users means None, not a fabricated zero score.
    """
    k = _valid_k(k)
    if not isinstance(ground_truth, Mapping) or not isinstance(predictions, Mapping):
        raise TypeError("ground_truth and predictions must be mappings keyed by raw user IDs")
    considered = _user_ids(ground_truth.keys() if user_ids is None else user_ids)
    per_user = []
    excluded = []
    for user_id in considered:
        relevant = set(_programme_ids(ground_truth.get(user_id, ()), label=f"ground_truth[{user_id!r}]"))
        if not relevant:
            excluded.append(user_id)
            continue
        if user_id not in predictions:
            raise ValueError(f"Missing predictions for evaluable user {user_id!r}")
        row = ranking_metrics_at_k(relevant, predictions[user_id], k)
        per_user.append({"user_id": user_id, **row})
    count = len(per_user)
    return {
        "k": k,
        "map": sum(row["ap"] for row in per_user) / count if count else None,
        "precision": sum(row["precision"] for row in per_user) / count if count else None,
        "recall": sum(row["recall"] for row in per_user) / count if count else None,
        "considered_users": len(considered),
        "evaluated_users": count,
        "excluded_no_relevant_count": len(excluded),
        "excluded_no_relevant_users": excluded,
        "evaluated_user_ids": [row["user_id"] for row in per_user],
        "per_user": per_user,
        "total_hits": sum(row["hits"] for row in per_user),
        "total_relevant": sum(row["relevant_count"] for row in per_user),
    }


def map_at_k(ground_truth: Mapping, predictions: Mapping, user_ids=None, k: int = 5) -> float | None:
    """Scalar convenience wrapper; use evaluate_rankings to retain cohort counts."""
    return evaluate_rankings(ground_truth, predictions, user_ids, k)["map"]


def stable_topk(scores: Iterable[float], candidate_ids: Iterable[int], k: int = 5) -> list[int]:
    """Return up to K IDs, by descending score and then ascending raw item ID.

    The score sequence must align one-to-one with unique positive candidate IDs.
    NaN, infinity, bools and nonnumeric scores are errors, not tie-breaking
    signals. Filter excluded candidates before calling; do not mask with -inf.
    An empty candidate sequence returns an empty list. This helper never fills
    a short list; the common fallback must be applied by the caller.
    """
    k = _valid_k(k)
    ids = _programme_ids(candidate_ids, label="candidate_ids")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids contains duplicate programme IDs")
    if isinstance(scores, (str, bytes, Mapping)) or scores is None:
        raise TypeError("scores must be a one-dimensional numeric iterable aligned to candidate_ids")
    values = list(scores)
    if len(values) != len(ids):
        raise ValueError(f"scores length {len(values)} does not match candidate_ids length {len(ids)}")
    clean_scores = []
    for index, score in enumerate(values):
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
            raise ValueError(f"scores[{index}] must be finite and numeric, got {score!r}")
        clean_scores.append(float(score))
    pairs = sorted(zip(ids, clean_scores), key=lambda pair: (-pair[1], pair[0]))
    return [programme_id for programme_id, _ in pairs[:k]]


def validate_ranked_lists(
    user_ids: Iterable[Hashable],
    predictions: Mapping[Hashable, Iterable[int]],
    candidate_id_set: Iterable[int],
    k: int = 5,
) -> dict[str, Any]:
    """Require one exact, duplicate-free K-list for every expected raw user ID.

    Reject missing/extra users, invalid IDs, programme 0, duplicate candidates,
    duplicate output items, and items outside the shared candidate pool. Order
    is retained, never sorted or repaired. Returns useful validation counts.
    A CSV-specific reader must additionally check the sample's exact header,
    column order and test-user row order once the actual sample is available.
    """
    k = _valid_k(k)
    expected = _user_ids(user_ids)
    if not expected:
        raise ValueError("Expected user_ids is empty; refusing to validate an empty submission")
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping keyed by the expected raw user IDs")
    pool_list = _programme_ids(candidate_id_set, label="candidate_id_set")
    pool = set(pool_list)
    if len(pool) != len(pool_list):
        raise ValueError("candidate_id_set contains duplicate programme IDs")
    if len(pool) < k:
        raise ValueError(f"Candidate pool has only {len(pool)} distinct items, fewer than k={k}")
    supplied = _user_ids(predictions.keys())
    expected_set = set(expected)
    missing = [user for user in expected if user not in predictions]
    extra = [user for user in supplied if user not in expected_set]
    if missing or extra:
        raise ValueError(f"User mismatch: missing {len(missing)} {missing[:10]!r}; extra {len(extra)} {extra[:10]!r}")
    recommended: set[int] = set()
    for user_id in expected:
        items = _programme_ids(predictions[user_id], label=f"predictions[{user_id!r}]")
        if len(items) != k:
            raise ValueError(f"User {user_id!r}: expected exactly {k} programme IDs, found {len(items)}")
        if len(set(items)) != k:
            raise ValueError(f"User {user_id!r}: repeated programme ID in ranked list {items!r}")
        unknown = [item for item in items if item not in pool]
        if unknown:
            raise ValueError(f"User {user_id!r}: programme IDs outside candidate pool: {unknown!r}")
        recommended.update(items)
    return {
        "valid": True,
        "users": len(expected),
        "k": k,
        "ranked_items": len(expected) * k,
        "candidate_items": len(pool),
        "distinct_recommended_items": len(recommended),
        "first_user": expected[0],
        "last_user": expected[-1],
    }


def validate_submission_csv(submission_path, sample_path, candidate_id_set, k: int = 5) -> dict[str, Any]:
    """Check an existing CSV against the ACTUAL sample, without rewriting it.

    The supplied SWEET.TV sample must have exactly ``user_id,tv_show_id`` in
    that order. User IDs are read as raw strings, preserving unsigned-64-bit
    values. The output must retain the exact sample row order and contain K
    canonical positive integer IDs separated by whitespace in every cell.
    Returns the hash of the exact validated output for score/artifact linkage.
    """
    k = _valid_k(k)
    def read(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise ValueError(f"CSV is empty: {path}")
        return rows[0], rows[1:]
    sample_header, sample_rows = read(sample_path)
    expected_header = ["user_id", "tv_show_id"]
    if sample_header != expected_header:
        raise ValueError(f"Unexpected actual sample header: {sample_header!r}; expected {expected_header!r}")
    header, rows = read(submission_path)
    if header != sample_header:
        raise ValueError(f"Submission header/order mismatch: {header!r}; actual sample is {sample_header!r}")
    if any(len(row) != 2 for row in sample_rows):
        raise ValueError("Actual sample has a malformed row; expected exactly two cells")
    if any(len(row) != 2 for row in rows):
        raise ValueError("Submission has a malformed row; expected exactly two cells")
    expected_users = [row[0] for row in sample_rows]
    observed_users = [row[0] for row in rows]
    if observed_users != expected_users:
        first_difference = next((i for i, pair in enumerate(zip(observed_users, expected_users)) if pair[0] != pair[1]), min(len(rows), len(sample_rows)))
        raise ValueError(f"Submission user row order/count differs from sample at row {first_difference + 1}; found {len(rows)}, expected {len(sample_rows)}")
    predictions = {}
    for user_id, cell in rows:
        tokens = cell.split()
        if any(re.fullmatch(r"[1-9][0-9]*", token) is None for token in tokens):
            raise ValueError(f"User {user_id!r}: programme IDs must be canonical positive decimal integers; got {cell!r}")
        predictions[user_id] = [int(token) for token in tokens]
    result = validate_ranked_lists(expected_users, predictions, candidate_id_set, k)
    raw = Path(submission_path).read_bytes()
    return {**result, "columns": header, "row_order_matches_sample": True,
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


ap_at_k = average_precision_at_k
