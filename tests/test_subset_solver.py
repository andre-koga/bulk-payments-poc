from bulk_payments.models import Bill, SubsetCandidate
from bulk_payments.retrieval import RetrievedBill
from bulk_payments.subset_solver import find_feasible_subsets, rank_subsets


def _rb(bid: str, amt: int) -> RetrievedBill:
    b = Bill(
        bill_id=bid,
        tenant_id="t",
        vendor_id="v",
        open_amount_minor=amt,
        currency="USD",
        open_date=__import__("datetime").date(2025, 1, 1),
    )
    return RetrievedBill(bill=b, vendor_link_score=1.0, date_delta_days=0)


def test_brute_force_unique_subset():
    cands = (_rb("a", 100), _rb("b", 200), _rb("c", 301))
    subs = find_feasible_subsets(cands, target_minor=301, tolerance_minor=0, max_solutions=20)
    assert len(subs) == 1
    assert subs[0].bill_ids == ("c",)


def test_ambiguous_two_subsets():
    cands = (_rb("a", 100), _rb("b", 200), _rb("c", 150), _rb("d", 150))
    subs = find_feasible_subsets(cands, target_minor=300, tolerance_minor=0, max_solutions=20)
    ids = {frozenset(s.bill_ids) for s in subs}
    assert frozenset({"a", "b"}) in ids
    assert frozenset({"c", "d"}) in ids


def test_mitm_path():
    cands = tuple(_rb(str(i), 1) for i in range(24))
    # sum 1*24 = 24, target 12 -> pick 12 ones - many solutions; just ensure runs
    subs = find_feasible_subsets(cands, target_minor=12, tolerance_minor=0, max_solutions=30)
    assert len(subs) >= 1


def test_rank_subsets_stable():
    s = (
        SubsetCandidate(("b", "a"), 10, 1),
        SubsetCandidate(("a", "b"), 10, 0),
    )
    r = rank_subsets(s)
    assert r[0].abs_residual_minor <= r[1].abs_residual_minor
