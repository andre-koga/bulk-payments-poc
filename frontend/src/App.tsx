import { useEffect, useMemo, useReducer, useState } from "react";
import {
  api,
  type BatchAutoMatchResponse,
  type MatchResult,
  type Payment,
  type Tenant,
} from "./api";
import "./App.css";

type MatchState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "result"; data: MatchResult }
  | { kind: "error"; message: string };

/** Per-payment review state (not keyed by event_id — each Run match starts a new event). */
type PaymentFeedback =
  | { status: "pending"; eventId: string }
  | { status: "top_rejected"; eventId: string }
  | { status: "accepted"; eventId: string; subsetIndex: number; billIds: string[] };

interface State {
  tenants: Tenant[];
  selectedTenant: string | null;
  payments: Payment[];
  match: Record<string, MatchState>;
  feedback: Record<string, PaymentFeedback>;
}

type Action =
  | { type: "SET_TENANTS"; tenants: Tenant[] }
  | { type: "SELECT_TENANT"; id: string }
  | { type: "SET_PAYMENTS"; payments: Payment[] }
  | { type: "MATCH_LOADING"; paymentId: string }
  | { type: "MATCH_RESULT"; paymentId: string; data: MatchResult }
  | { type: "MATCH_ERROR"; paymentId: string; message: string }
  | { type: "TOP_REJECTED"; paymentId: string; eventId: string }
  | { type: "ACCEPTED"; paymentId: string; eventId: string; subsetIndex: number; billIds: string[] }
  | { type: "CLEAR_FEEDBACK"; paymentId: string }
  | { type: "RESET_PAYMENT"; paymentId: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_TENANTS":
      return { ...state, tenants: action.tenants };
    case "SELECT_TENANT":
      return {
        ...state,
        selectedTenant: action.id,
        payments: [],
        match: {},
        feedback: {},
      };
    case "SET_PAYMENTS":
      return { ...state, payments: action.payments };
    case "MATCH_LOADING":
      return { ...state, match: { ...state.match, [action.paymentId]: { kind: "loading" } } };
    case "MATCH_RESULT": {
      const eventId = action.data.event_id;
      const nextFeedback = { ...state.feedback };
      if (eventId) {
        nextFeedback[action.paymentId] = { status: "pending", eventId };
      }
      return {
        ...state,
        match: { ...state.match, [action.paymentId]: { kind: "result", data: action.data } },
        feedback: nextFeedback,
      };
    }
    case "MATCH_ERROR":
      return { ...state, match: { ...state.match, [action.paymentId]: { kind: "error", message: action.message } } };
    case "TOP_REJECTED":
      return {
        ...state,
        feedback: {
          ...state.feedback,
          [action.paymentId]: { status: "top_rejected", eventId: action.eventId },
        },
      };
    case "ACCEPTED":
      return {
        ...state,
        feedback: {
          ...state.feedback,
          [action.paymentId]: {
            status: "accepted",
            eventId: action.eventId,
            subsetIndex: action.subsetIndex,
            billIds: action.billIds,
          },
        },
      };
    case "CLEAR_FEEDBACK": {
      const next = { ...state.feedback };
      delete next[action.paymentId];
      return { ...state, feedback: next };
    }
    case "RESET_PAYMENT": {
      const nextFb = { ...state.feedback };
      delete nextFb[action.paymentId];
      const nextMatch = { ...state.match };
      delete nextMatch[action.paymentId];
      return { ...state, feedback: nextFb, match: nextMatch };
    }
    default:
      return state;
  }
}

const initialState: State = {
  tenants: [],
  selectedTenant: null,
  payments: [],
  match: {},
  feedback: {},
};

function decisionBadge(decision: MatchResult["decision"]) {
  const map: Record<string, { label: string; cls: string }> = {
    auto_applied: { label: "Auto-applied", cls: "badge badge-auto" },
    suggested: { label: "Suggestion", cls: "badge badge-suggest" },
    no_candidates: { label: "No match", cls: "badge badge-none" },
    rejected_by_user: { label: "Rejected", cls: "badge badge-none" },
    manual: { label: "Manual", cls: "badge badge-none" },
  };
  const d = map[decision] ?? { label: decision, cls: "badge" };
  return <span className={d.cls}>{d.label}</span>;
}

function billIdsKey(ids: string[]) {
  return [...ids].sort().join(",");
}

type PendingReviewAction =
  | { kind: "accept_top"; eventId: string; billIds: string[] }
  | { kind: "accept_alt"; eventId: string; subsetIndex: number; billIds: string[] }
  | { kind: "reject_top"; eventId: string };

function needsExpertReasoning(data: MatchResult) {
  return data.decision === "suggested" || data.competing_subset_count > 1;
}

function AutoMatchBar({
  loading,
  summary,
  openCount,
  onRun,
}: {
  loading: boolean;
  summary: BatchAutoMatchResponse["summary"] | null;
  openCount: number;
  onRun: () => void;
}) {
  return (
    <section className="auto-match-bar">
      <div className="auto-match-copy">
        <h2>Autonomous matching</h2>
        <p>
          Run the matcher across all open payments. Confident unique matches are applied
          automatically; ambiguous cases are queued below for expert review.
        </p>
      </div>
      <button
        type="button"
        className="btn-auto-match"
        onClick={onRun}
        disabled={loading || openCount === 0}
      >
        {loading ? "Running auto-match…" : `Auto-match all (${openCount} open)`}
      </button>
      {summary && (
        <div className="auto-match-stats">
          <span className="stat stat-auto">{summary.auto_matched} auto-matched</span>
          <span className="stat stat-review">{summary.needs_review} expert review</span>
          <span className="stat stat-none">{summary.no_match} no match</span>
          {summary.skipped_fully_allocated > 0 && (
            <span className="stat stat-skip">
              {summary.skipped_fully_allocated} already complete
            </span>
          )}
        </div>
      )}
    </section>
  );
}

function MatchPanel({
  payment,
  matchState,
  paymentFeedback,
  expertReview,
  onRunMatch,
  onAcceptTop,
  onRejectTop,
  onAcceptAlternate,
  onUnmatch,
  unmatchLoading,
}: {
  payment: Payment;
  matchState: MatchState;
  paymentFeedback: PaymentFeedback | undefined;
  expertReview: boolean;
  onRunMatch: () => void;
  onAcceptTop: (eventId: string, billIds: string[], reasoning?: string) => void;
  onRejectTop: (eventId: string, reasoning?: string) => void;
  onAcceptAlternate: (
    eventId: string,
    subsetIndex: number,
    billIds: string[],
    reasoning?: string
  ) => void;
  onUnmatch: () => void;
  unmatchLoading: boolean;
}) {
  const [pendingAction, setPendingAction] = useState<PendingReviewAction | null>(null);
  const [reasoning, setReasoning] = useState("");
  const [reasoningError, setReasoningError] = useState<string | null>(null);
  const matchedBillIds =
    paymentFeedback?.status === "accepted"
      ? paymentFeedback.billIds
      : (payment.settlement?.bill_ids ?? []);
  const hasAllocation = matchedBillIds.length > 0;
  const fullyAllocated =
    payment.settlement?.is_fully_allocated === true ||
    (matchState.kind === "result" && matchState.data.is_fully_allocated);
  const allocatedMinor =
    payment.settlement?.allocated_sum_minor ??
    (matchState.kind === "result" ? matchState.data.allocated_sum_minor : 0);
  const remainingMinor =
    payment.settlement?.remaining_minor ??
    (matchState.kind === "result" ? matchState.data.remaining_minor : payment.amount_minor);

  const topRejected = paymentFeedback?.status === "top_rejected";
  const eventId =
    matchState.kind === "result" ? matchState.data.event_id : paymentFeedback?.eventId;

  const canReviewMatch =
    matchState.kind === "result" &&
    !fullyAllocated &&
    matchState.data.subsets.length > 0 &&
    !matchState.data.reason_codes.includes("payment_fully_allocated");

  const matchData = matchState.kind === "result" ? matchState.data : null;
  const expertReasoningRequired = matchData ? needsExpertReasoning(matchData) : false;

  const submitPendingAction = () => {
    if (!pendingAction) return;
    if (expertReasoningRequired && !reasoning.trim()) {
      setReasoningError("Please explain your choice — used to train the ranker.");
      return;
    }
    const text = reasoning.trim() || undefined;
    if (pendingAction.kind === "accept_top") {
      onAcceptTop(pendingAction.eventId, pendingAction.billIds, text);
    } else if (pendingAction.kind === "accept_alt") {
      onAcceptAlternate(
        pendingAction.eventId,
        pendingAction.subsetIndex,
        pendingAction.billIds,
        text
      );
    } else {
      onRejectTop(pendingAction.eventId, text);
    }
    setPendingAction(null);
    setReasoning("");
    setReasoningError(null);
  };

  const requestAcceptTop = (eid: string, billIds: string[]) => {
    if (expertReasoningRequired) {
      setPendingAction({ kind: "accept_top", eventId: eid, billIds });
      setReasoning("");
      setReasoningError(null);
      return;
    }
    onAcceptTop(eid, billIds);
  };

  const requestAcceptAlt = (eid: string, subsetIndex: number, billIds: string[]) => {
    if (expertReasoningRequired) {
      setPendingAction({ kind: "accept_alt", eventId: eid, subsetIndex, billIds });
      setReasoning("");
      setReasoningError(null);
      return;
    }
    onAcceptAlternate(eid, subsetIndex, billIds);
  };

  const requestRejectTop = (eid: string) => {
    if (expertReasoningRequired) {
      setPendingAction({ kind: "reject_top", eventId: eid });
      setReasoning("");
      setReasoningError(null);
      return;
    }
    onRejectTop(eid);
  };

  return (
    <div className={`match-panel ${expertReview ? "match-panel-expert" : ""}`}>
      {expertReview && (
        <div className="expert-review-banner">
          <span className="badge badge-suggest">Expert review</span>
          Ambiguous match — choose a subset and document why.
        </div>
      )}
      <div className="payment-row">
        <div className="payment-meta">
          <span className="payment-amount">{payment.amount_display}</span>
          <span className="payment-date">{payment.payment_date}</span>
          <span className="payment-counterparty">{payment.counterparty_bank_name}</span>
          {payment.description && (
            <span className="payment-desc">"{payment.description}"</span>
          )}
        </div>
        <button
          className="btn-match"
          onClick={onRunMatch}
          disabled={matchState.kind === "loading" || fullyAllocated}
          title={
            fullyAllocated
              ? "Payment amount is already fully covered by allocated bills"
              : undefined
          }
        >
          {matchState.kind === "loading"
            ? "Matching…"
            : hasAllocation
              ? "Match remaining amount"
              : "Run match"}
        </button>
      </div>

      {hasAllocation && (
        <div className="settled-banner">
          <span className="badge badge-auto">
            {fullyAllocated ? "Fully matched" : "Partially matched"}
          </span>
          <span className="settled-label">Allocated:</span>
          {matchedBillIds.map((id) => (
            <code key={id} className="bill-chip">{id}</code>
          ))}
          <span className="settled-hint">
            ${(allocatedMinor / 100).toFixed(2)} of {payment.amount_display} allocated.
            {fullyAllocated
              ? " This payment is complete — no further bills can be added."
              : remainingMinor != null && remainingMinor > 0
                ? ` ~$${(remainingMinor / 100).toFixed(2)} remaining to match among open bills.`
                : ""}
          </span>
          <button
            type="button"
            className="btn-unmatch"
            onClick={onUnmatch}
            disabled={unmatchLoading || matchState.kind === "loading"}
          >
            {unmatchLoading ? "Unmatching…" : "Unmatch — reset payment"}
          </button>
        </div>
      )}

      {matchState.kind === "error" && (
        <p className="error-msg">Error: {matchState.message}</p>
      )}

      {matchState.kind === "result" && (
        <div className="result-area">
          <div className="result-header">
            {decisionBadge(matchState.data.decision)}
            {matchState.data.competing_subset_count > 1 && canReviewMatch && (
              <span className="competing-note">
                {matchState.data.competing_subset_count} possible subsets
              </span>
            )}
            {matchState.data.calibrated_accept_prob !== null && (
              <span className="prob-note">
                p(accept)={((matchState.data.calibrated_accept_prob ?? 0) * 100).toFixed(1)}%
              </span>
            )}
          </div>

          {expertReasoningRequired && canReviewMatch && (
            <p className="alt-hint">
              Multiple valid subsets — your reasoning is saved for model fine-tuning.
            </p>
          )}

          {pendingAction && (
            <div className="reasoning-panel">
              <label className="reasoning-label" htmlFor={`reasoning-${payment.payment_id}`}>
                {pendingAction.kind === "reject_top"
                  ? "Why reject the top suggestion?"
                  : "Why did you choose this subset?"}
              </label>
              <textarea
                id={`reasoning-${payment.payment_id}`}
                className="reasoning-input"
                rows={3}
                placeholder="e.g. Vendor alias matches bank statement; PO #12345 on remittance…"
                value={reasoning}
                onChange={(e) => {
                  setReasoning(e.target.value);
                  setReasoningError(null);
                }}
              />
              {reasoningError && <p className="reasoning-error">{reasoningError}</p>}
              <div className="feedback-row">
                <button type="button" className="btn-accept" onClick={submitPendingAction}>
                  Confirm
                </button>
                <button
                  type="button"
                  className="btn-reject"
                  onClick={() => {
                    setPendingAction(null);
                    setReasoning("");
                    setReasoningError(null);
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {topRejected && canReviewMatch && (
            <p className="alt-hint">
              Top suggestion rejected — choose another subset below.
            </p>
          )}

          {hasAllocation && !fullyAllocated && canReviewMatch && remainingMinor != null && (
            <p className="alt-hint">
              Suggestions below close the remaining ~${(remainingMinor / 100).toFixed(2)}, not the full payment again.
            </p>
          )}

          {matchState.data.reason_codes.includes("payment_fully_allocated") && (
            <p className="no-subsets">
              This payment is already fully allocated. Use <strong>Unmatch</strong> above to clear bills and match again.
            </p>
          )}

          {matchState.data.reason_codes.length > 0 && (
            <div className="reason-codes">
              {matchState.data.reason_codes.map((r) => (
                <code key={r} className="reason-chip">{r}</code>
              ))}
            </div>
          )}

          {matchState.data.subsets.length === 0 &&
            !matchState.data.reason_codes.includes("payment_fully_allocated") && (
            <p className="no-subsets">
              {hasAllocation
                ? "No open bills match the remaining amount."
                : "No feasible bill subsets found for this payment amount."}
            </p>
          )}

          {canReviewMatch && matchState.data.subsets.length > 1 && (
            <p className="subsets-scroll-hint">
              {matchState.data.subsets.length} suggestions shown
              {matchState.data.competing_subset_count > matchState.data.subsets.length
                ? ` (${matchState.data.competing_subset_count} total)`
                : ""}
              — scroll inside the panel
            </p>
          )}

          {canReviewMatch && (
            <div
              className={
                matchState.data.subsets.length > 1 ? "subsets-scroll" : "subsets-list"
              }
              role="region"
              aria-label="Match suggestions"
            >
            {matchState.data.subsets.map((subset, i) => {
            const isTop = i === 0;
            const isAcceptedChoice =
              hasAllocation &&
              billIdsKey(matchedBillIds) === billIdsKey(subset.bill_ids) &&
              (paymentFeedback?.status !== "accepted" || paymentFeedback.subsetIndex === i);

            const showTopActions =
              canReviewMatch && isTop && eventId && paymentFeedback?.status === "pending";
            const showAltAccept = canReviewMatch && !isTop && topRejected && eventId;

            return (
              <div
                key={billIdsKey(subset.bill_ids)}
                className={`subset-card ${isTop && !topRejected ? "subset-best" : ""} ${isAcceptedChoice ? "subset-chosen" : ""}`}
              >
                <div className="subset-header">
                  <span className="subset-rank">#{i + 1}</span>
                  <span className="subset-sum">{subset.sum_display}</span>
                  {subset.abs_residual_minor > 0 && (
                    <span className="residual">Δ ${(subset.abs_residual_minor / 100).toFixed(2)}</span>
                  )}
                </div>
                <ul className="bill-list">
                  {subset.bill_ids.map((bid) => (
                    <li key={bid} className="bill-chip">{bid}</li>
                  ))}
                </ul>

                {showTopActions && !pendingAction && (
                  <div className="feedback-row">
                    <button
                      className="btn-accept"
                      onClick={() => requestAcceptTop(eventId!, subset.bill_ids)}
                    >
                      ✓ Accept
                    </button>
                    <button
                      className="btn-reject"
                      onClick={() => requestRejectTop(eventId!)}
                    >
                      ✗ Reject
                    </button>
                  </div>
                )}

                {isTop && topRejected && canReviewMatch && (
                  <div className="feedback-done feedback-rejected">Top suggestion rejected ✗</div>
                )}

                {showAltAccept && !pendingAction && (
                  <div className="feedback-row">
                    <button
                      className="btn-accept"
                      onClick={() => requestAcceptAlt(eventId!, i, subset.bill_ids)}
                    >
                      ✓ Accept this subset
                    </button>
                  </div>
                )}

                {isAcceptedChoice && (
                  <div className="feedback-done feedback-accepted">Accepted ✓</div>
                )}
              </div>
            );
          })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [apiError, setApiError] = useState<string | null>(null);
  const [unmatchLoadingId, setUnmatchLoadingId] = useState<string | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchSummary, setBatchSummary] = useState<BatchAutoMatchResponse["summary"] | null>(
    null
  );
  const [expertReviewIds, setExpertReviewIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    api
      .listTenants()
      .then((ts) => {
        dispatch({ type: "SET_TENANTS", tenants: ts });
        if (ts.length > 0) dispatch({ type: "SELECT_TENANT", id: ts[0].tenant_id });
      })
      .catch((e) => setApiError(String(e)));
  }, []);

  useEffect(() => {
    if (!state.selectedTenant) return;
    setBatchSummary(null);
    setExpertReviewIds(new Set());
    api
      .listPayments(state.selectedTenant)
      .then((ps) => dispatch({ type: "SET_PAYMENTS", payments: ps }))
      .catch((e) => setApiError(String(e)));
  }, [state.selectedTenant]);

  const openPaymentCount = useMemo(
    () => state.payments.filter((p) => !p.settlement?.is_fully_allocated).length,
    [state.payments]
  );

  const sortedPayments = useMemo(() => {
    return [...state.payments].sort((a, b) => {
      const aExpert = expertReviewIds.has(a.payment_id) ? 0 : 1;
      const bExpert = expertReviewIds.has(b.payment_id) ? 0 : 1;
      if (aExpert !== bExpert) return aExpert - bExpert;
      return a.payment_date.localeCompare(b.payment_date);
    });
  }, [state.payments, expertReviewIds]);

  const isFullyAllocated = (payment: Payment) =>
    payment.settlement?.is_fully_allocated === true;

  const refreshPayments = () => {
    if (!state.selectedTenant) return Promise.resolve();
    return api
      .listPayments(state.selectedTenant)
      .then((ps) => dispatch({ type: "SET_PAYMENTS", payments: ps }));
  };

  const handleRunMatch = (payment: Payment) => {
    if (!state.selectedTenant) return;
    if (isFullyAllocated(payment)) {
      return;
    }
    if (payment.settlement && !payment.settlement.is_fully_allocated) {
      const ok = window.confirm(
        `~$${(payment.settlement.remaining_minor / 100).toFixed(2)} of this payment is still unallocated. ` +
          "Match again only searches open bills for that remaining amount."
      );
      if (!ok) return;
      dispatch({ type: "CLEAR_FEEDBACK", paymentId: payment.payment_id });
    }

    dispatch({ type: "MATCH_LOADING", paymentId: payment.payment_id });
    api
      .runMatch(state.selectedTenant, payment.payment_id)
      .then((r) => dispatch({ type: "MATCH_RESULT", paymentId: payment.payment_id, data: r }))
      .catch((e) =>
        dispatch({ type: "MATCH_ERROR", paymentId: payment.payment_id, message: String(e) })
      );
  };

  const handleAcceptTop = (
    paymentId: string,
    eventId: string,
    billIds: string[],
    reasoning?: string
  ) => {
    api
      .recordOutcome(eventId, "accepted_as_is", billIds, { accountantReasoning: reasoning })
      .then(() => {
        dispatch({ type: "ACCEPTED", paymentId, eventId, subsetIndex: 0, billIds });
        setExpertReviewIds((prev) => {
          const next = new Set(prev);
          next.delete(paymentId);
          return next;
        });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)));
  };

  const handleRejectTop = (paymentId: string, eventId: string, reasoning?: string) => {
    api
      .recordOutcome(eventId, "rejected", undefined, { accountantReasoning: reasoning })
      .then(() => dispatch({ type: "TOP_REJECTED", paymentId, eventId }))
      .catch((e) => setApiError(String(e)));
  };

  const handleAcceptAlternate = (
    paymentId: string,
    eventId: string,
    subsetIndex: number,
    billIds: string[],
    reasoning?: string
  ) => {
    api
      .recordOutcome(eventId, "edited_subset", billIds, { accountantReasoning: reasoning })
      .then(() => {
        dispatch({ type: "ACCEPTED", paymentId, eventId, subsetIndex, billIds });
        setExpertReviewIds((prev) => {
          const next = new Set(prev);
          next.delete(paymentId);
          return next;
        });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)));
  };

  const handleAutoMatchAll = () => {
    if (!state.selectedTenant) return;
    setBatchLoading(true);
    setApiError(null);
    api
      .autoMatchAll(state.selectedTenant)
      .then((res) => {
        setBatchSummary(res.summary);
        const reviewIds = new Set(res.needs_review.map((m) => m.payment_id));
        setExpertReviewIds(reviewIds);
        res.needs_review.forEach((m) => {
          dispatch({ type: "MATCH_RESULT", paymentId: m.payment_id, data: m });
        });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)))
      .finally(() => setBatchLoading(false));
  };

  const handleUnmatch = (payment: Payment) => {
    if (!state.selectedTenant) return;
    const n = payment.settlement?.bill_ids.length ?? 0;
    const ok = window.confirm(
      `Clear all ${n} bill allocation(s) for this payment? ` +
        "The payment will be unmatched and you can run match again from scratch."
    );
    if (!ok) return;

    setUnmatchLoadingId(payment.payment_id);
    api
      .unmatchPayment(state.selectedTenant, payment.payment_id)
      .then(() => {
        dispatch({ type: "RESET_PAYMENT", paymentId: payment.payment_id });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)))
      .finally(() => setUnmatchLoadingId(null));
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Bulk Payment Matcher</h1>
        {state.tenants.length > 1 && (
          <select
            className="tenant-select"
            value={state.selectedTenant ?? ""}
            onChange={(e) => dispatch({ type: "SELECT_TENANT", id: e.target.value })}
          >
            {state.tenants.map((t) => (
              <option key={t.tenant_id} value={t.tenant_id}>
                {t.tenant_id} ({t.ledger_currency})
              </option>
            ))}
          </select>
        )}
        {state.selectedTenant && state.tenants.length === 1 && (
          <span className="tenant-label">Tenant: {state.selectedTenant}</span>
        )}
      </header>

      {apiError && (
        <div className="api-error">
          API error: {apiError} — is the backend running?{" "}
          <button type="button" onClick={() => setApiError(null)}>✕</button>
        </div>
      )}

      {state.selectedTenant && state.payments.length > 0 && (
        <AutoMatchBar
          loading={batchLoading}
          summary={batchSummary}
          openCount={openPaymentCount}
          onRun={handleAutoMatchAll}
        />
      )}

      <main className="payment-list">
        {state.payments.length === 0 && !apiError && (
          <p className="empty-state">
            No payments found. Run <code>bulk-match seed</code> to load demo data.
          </p>
        )}
        {sortedPayments.map((payment) => (
          <MatchPanel
            key={payment.payment_id}
            payment={payment}
            matchState={state.match[payment.payment_id] ?? { kind: "idle" }}
            paymentFeedback={state.feedback[payment.payment_id]}
            expertReview={expertReviewIds.has(payment.payment_id)}
            onRunMatch={() => handleRunMatch(payment)}
            onAcceptTop={(eid, bills, reasoning) =>
              handleAcceptTop(payment.payment_id, eid, bills, reasoning)
            }
            onRejectTop={(eid, reasoning) =>
              handleRejectTop(payment.payment_id, eid, reasoning)
            }
            onAcceptAlternate={(eid, idx, bills, reasoning) =>
              handleAcceptAlternate(payment.payment_id, eid, idx, bills, reasoning)
            }
            onUnmatch={() => handleUnmatch(payment)}
            unmatchLoading={unmatchLoadingId === payment.payment_id}
          />
        ))}
      </main>
    </div>
  );
}
