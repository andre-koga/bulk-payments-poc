from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import combinations
from typing import Iterable

from bulk_payments.models import SubsetCandidate
from bulk_payments.retrieval import RetrievedBill


def find_feasible_subsets(
    candidates: tuple[RetrievedBill, ...],
    target_minor: int,
    tolerance_minor: int,
    *,
    max_solutions: int = 50,
) -> tuple[SubsetCandidate, ...]:
    """
    Enumerate subsets S where |sum(S) - target| <= tolerance.
    Uses brute force for n <= 22 else meet-in-the-middle (exact for integers).
    """
    amounts = [rb.bill.open_amount_minor for rb in candidates]
    ids = [rb.bill.bill_id for rb in candidates]
    n = len(amounts)
    if n == 0:
        return ()

    if n <= 22:
        return _brute_force(ids, amounts, target_minor, tolerance_minor, max_solutions=max_solutions)
    return _meet_in_middle(ids, amounts, target_minor, tolerance_minor, max_solutions=max_solutions)


def _brute_force(
    ids: list[str],
    amounts: list[int],
    target: int,
    tol: int,
    *,
    max_solutions: int,
) -> tuple[SubsetCandidate, ...]:
    out: list[SubsetCandidate] = []
    n = len(ids)
    for r in range(1, n + 1):
        for idxs in combinations(range(n), r):
            s = sum(amounts[i] for i in idxs)
            res = abs(s - target)
            if res <= tol:
                bids = tuple(ids[i] for i in idxs)
                out.append(SubsetCandidate(bill_ids=bids, sum_minor=s, abs_residual_minor=res))
                if len(out) >= max_solutions:
                    return tuple(out)
    return tuple(out)


def _meet_in_middle(
    ids: list[str],
    amounts: list[int],
    target: int,
    tol: int,
    *,
    max_solutions: int,
) -> tuple[SubsetCandidate, ...]:
    n = len(amounts)
    mid = n // 2
    left = amounts[:mid]
    right = amounts[mid:]
    left_ids = ids[:mid]
    right_ids = ids[mid:]

    left_sums: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for r in range(len(left) + 1):
        for idxs in combinations(range(len(left)), r):
            sm = sum(left[i] for i in idxs)
            bid = tuple(left_ids[i] for i in idxs)
            left_sums[sm].append(bid)

    right_list: list[tuple[int, tuple[str, ...]]] = []
    for r in range(len(right) + 1):
        for idxs in combinations(range(len(right)), r):
            sm = sum(right[i] for i in idxs)
            bid = tuple(right_ids[i] for i in idxs)
            right_list.append((sm, bid))
    right_list.sort(key=lambda x: x[0])
    right_sums = [x[0] for x in right_list]

    out: list[SubsetCandidate] = []
    for lsum, lbids_list in left_sums.items():
        lo = target - lsum - tol
        hi = target - lsum + tol
        i = bisect_left(right_sums, lo)
        j = bisect_right(right_sums, hi)
        for rsum, rbids in right_list[i:j]:
            total = lsum + rsum
            res = abs(total - target)
            if res > tol:
                continue
            for lb in lbids_list:
                merged = tuple(sorted(lb + rbids))
                out.append(SubsetCandidate(bill_ids=merged, sum_minor=total, abs_residual_minor=res))
                if len(out) >= max_solutions:
                    return tuple(out)
    return tuple(out)


def rank_subsets(subsets: Iterable[SubsetCandidate]) -> tuple[SubsetCandidate, ...]:
    """Prefer smaller residual, then fewer bills, then lexicographic bill ids for stability."""
    return tuple(
        sorted(
            subsets,
            key=lambda s: (s.abs_residual_minor, len(s.bill_ids), s.bill_ids),
        )
    )
