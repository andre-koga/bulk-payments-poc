import { useEffect, useReducer, useState } from "react";
import { api, type MatchResult, type Payment, type Tenant } from "./api";
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
      return { ...state, selectedTenant: action.id, payments: [], match: {}, feedback: {} };
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

function MatchPanel({
  payment,
  matchState,
  paymentFeedback,
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
  onRunMatch: () => void;
  onAcceptTop: (eventId: string, billIds: string[]) => void;
  onRejectTop: (eventId: string) => void;
  onAcceptAlternate: (eventId: string, subsetIndex: number, billIds: string[]) => void;
  onUnmatch: () => void;
  unmatchLoading: boolean;
}) {
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

  return (
    <div className="match-panel">
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

          {canReviewMatch &&
            matchState.data.subsets.map((subset, i) => {
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

                {showTopActions && (
                  <div className="feedback-row">
                    <button
                      className="btn-accept"
                      onClick={() => onAcceptTop(eventId!, subset.bill_ids)}
                    >
                      ✓ Accept
                    </button>
                    <button
                      className="btn-reject"
                      onClick={() => onRejectTop(eventId!)}
                    >
                      ✗ Reject
                    </button>
                  </div>
                )}

                {isTop && topRejected && canReviewMatch && (
                  <div className="feedback-done feedback-rejected">Top suggestion rejected ✗</div>
                )}

                {showAltAccept && (
                  <div className="feedback-row">
                    <button
                      className="btn-accept"
                      onClick={() => onAcceptAlternate(eventId!, i, subset.bill_ids)}
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
  );
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [apiError, setApiError] = useState<string | null>(null);
  const [unmatchLoadingId, setUnmatchLoadingId] = useState<string | null>(null);

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
    api
      .listPayments(state.selectedTenant)
      .then((ps) => dispatch({ type: "SET_PAYMENTS", payments: ps }))
      .catch((e) => setApiError(String(e)));
  }, [state.selectedTenant]);

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

  const handleAcceptTop = (paymentId: string, eventId: string, billIds: string[]) => {
    api
      .recordOutcome(eventId, "accepted_as_is", billIds)
      .then(() => {
        dispatch({ type: "ACCEPTED", paymentId, eventId, subsetIndex: 0, billIds });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)));
  };

  const handleRejectTop = (paymentId: string, eventId: string) => {
    api
      .recordOutcome(eventId, "rejected")
      .then(() => dispatch({ type: "TOP_REJECTED", paymentId, eventId }))
      .catch((e) => setApiError(String(e)));
  };

  const handleAcceptAlternate = (
    paymentId: string,
    eventId: string,
    subsetIndex: number,
    billIds: string[]
  ) => {
    api
      .recordOutcome(eventId, "edited_subset", billIds)
      .then(() => {
        dispatch({ type: "ACCEPTED", paymentId, eventId, subsetIndex, billIds });
        return refreshPayments();
      })
      .catch((e) => setApiError(String(e)));
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

      <main className="payment-list">
        {state.payments.length === 0 && !apiError && (
          <p className="empty-state">
            No payments found. Run <code>bulk-match seed</code> to load demo data.
          </p>
        )}
        {state.payments.map((payment) => (
          <MatchPanel
            key={payment.payment_id}
            payment={payment}
            matchState={state.match[payment.payment_id] ?? { kind: "idle" }}
            paymentFeedback={state.feedback[payment.payment_id]}
            onRunMatch={() => handleRunMatch(payment)}
            onAcceptTop={(eid, bills) => handleAcceptTop(payment.payment_id, eid, bills)}
            onRejectTop={(eid) => handleRejectTop(payment.payment_id, eid)}
            onAcceptAlternate={(eid, idx, bills) =>
              handleAcceptAlternate(payment.payment_id, eid, idx, bills)
            }
            onUnmatch={() => handleUnmatch(payment)}
            unmatchLoading={unmatchLoadingId === payment.payment_id}
          />
        ))}
      </main>
    </div>
  );
}
