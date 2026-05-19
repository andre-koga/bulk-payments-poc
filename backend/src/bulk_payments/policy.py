from __future__ import annotations

from dataclasses import dataclass

from bulk_payments.models import DecisionKind, Payment, SubsetCandidate, TenantConfig


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds for auto vs suggest. Cold start: ranker disabled uses structural rules only."""

    tau_auto: float = 0.99
    tau_suggest: float = 0.35
    min_vendor_link_auto: float = 0.55
    min_vendor_link_suggest: float = 0.25
    unique_subset_required_for_auto: bool = True
    min_best_second_residual_gap: int | None = None  # minor units; None = ignore
    require_ranker_for_auto: bool = False  # True once model is deployed
    # When False (cold start), ranker probabilities are logged but do not block structural auto-apply.
    use_ranker_threshold_for_auto: bool = False


def choose_decision(
    *,
    payment: Payment,
    cfg: TenantConfig,
    policy: PolicyConfig,
    feasible: tuple[SubsetCandidate, ...],
    unique_amount_closure: bool,
    competing_subset_count: int,
    best: SubsetCandidate | None,
    second: SubsetCandidate | None,
    vendor_link_best: float,
    calibrated_prob: float | None,
    ranker_score: float | None,
) -> tuple[DecisionKind, tuple[str, ...]]:
    reasons: list[str] = []

    if best is None or not feasible:
        return DecisionKind.NO_CANDIDATES, ("no_feasible_subset",)

    if vendor_link_best < policy.min_vendor_link_suggest:
        return DecisionKind.NO_CANDIDATES, ("weak_vendor_link",)

    # Suggest path when multiple competing subsets or not unique closure
    ambiguous = (not unique_amount_closure) or (competing_subset_count > 1)
    if ambiguous:
        reasons.append("ambiguous_subsets")

    gap_ok = True
    if policy.min_best_second_residual_gap is not None and second is not None:
        gap = second.abs_residual_minor - best.abs_residual_minor
        gap_ok = gap >= policy.min_best_second_residual_gap
        if not gap_ok:
            reasons.append("small_residual_gap_vs_runner_up")

    prob = calibrated_prob
    if policy.require_ranker_for_auto and prob is None:
        return DecisionKind.SUGGESTED, tuple(reasons + ["ranker_required_not_loaded"])

    if policy.use_ranker_threshold_for_auto:
        auto_prob_ok = prob is None or prob >= policy.tau_auto
        if prob is not None and prob < policy.tau_auto:
            reasons.append("ranker_below_tau_auto")
    else:
        auto_prob_ok = True
    suggest_prob_ok = prob is None or prob >= policy.tau_suggest

    if (
        not ambiguous
        and gap_ok
        and vendor_link_best >= policy.min_vendor_link_auto
        and auto_prob_ok
        and policy.unique_subset_required_for_auto
        and unique_amount_closure
    ):
        # Exposure caps
        if len(best.bill_ids) > cfg.max_auto_bill_count:
            return DecisionKind.SUGGESTED, tuple(reasons + ["exceeds_max_auto_bill_count"])
        if cfg.max_auto_amount_minor is not None and payment.amount_minor > cfg.max_auto_amount_minor:
            return DecisionKind.SUGGESTED, tuple(reasons + ["exceeds_max_auto_amount"])

        return DecisionKind.AUTO_APPLIED, ("unique_subset", "gates_ok", "vendor_link_ok")

    if suggest_prob_ok and vendor_link_best >= policy.min_vendor_link_suggest:
        return DecisionKind.SUGGESTED, tuple(reasons + ["below_auto_threshold_or_ambiguous"])

    return DecisionKind.NO_CANDIDATES, tuple(reasons + ["below_suggest_threshold"])
