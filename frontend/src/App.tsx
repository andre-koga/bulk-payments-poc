import { useEffect, useReducer, useState } from "react";
import { api, type MatchResult, type Payment, type Tenant } from "./api";
import "./App.css";

// ── state machine ──────────────────────────────────────────────────────────
type MatchState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "result"; data: MatchResult }
  | { kind: "error"; message: string };

type FeedbackState = Record<string, "accepted" | "rejected" | "pending">;

interface State {
  tenants: Tenant[];
  selectedTenant: string | null;
  payments: Payment[];
  match: Record<string, MatchState>;
  feedback: FeedbackState;
}

type Action =
  | { type: "SET_TENANTS"; tenants: Tenant[] }
  | { type: "SELECT_TENANT"; id: string }
  | { type: "SET_PAYMENTS"; payments: Payment[] }
  | { type: "MATCH_LOADING"; paymentId: string }
  | { type: "MATCH_RESULT"; paymentId: string; data: MatchResult }
  | { type: "MATCH_ERROR"; paymentId: string; message: string }
  | { type: "FEEDBACK"; eventId: string; status: "accepted" | "rejected" };

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
    case "MATCH_RESULT":
      return { ...state, match: { ...state.match, [action.paymentId]: { kind: "result", data: action.data } } };
    case "MATCH_ERROR":
      return { ...state, match: { ...state.match, [action.paymentId]: { kind: "error", message: action.message } } };
    case "FEEDBACK":
      return { ...state, feedback: { ...state.feedback, [action.eventId]: action.status } };
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

// ── helpers ────────────────────────────────────────────────────────────────

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

function fmtAmount(minor: number) {
  return `$${(minor / 100).toFixed(2)}`;
}

// ── sub-components ─────────────────────────────────────────────────────────

function MatchPanel({
  payment,
  matchState,
  feedback,
  onRunMatch,
  onAccept,
  onReject,
}: {
  payment: Payment;
  matchState: MatchState;
  feedback: FeedbackState;
  onRunMatch: () => void;
  onAccept: (eventId: string) => void;
  onReject: (eventId: string) => void;
}) {
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
          disabled={matchState.kind === "loading"}
        >
          {matchState.kind === "loading" ? "Matching…" : "Run match"}
        </button>
      </div>

      {matchState.kind === "error" && (
        <p className="error-msg">Error: {matchState.message}</p>
      )}

      {matchState.kind === "result" && (
        <div className="result-area">
          <div className="result-header">
            {decisionBadge(matchState.data.decision)}
            {matchState.data.competing_subset_count > 1 && (
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

          {matchState.data.reason_codes.length > 0 && (
            <div className="reason-codes">
              {matchState.data.reason_codes.map((r) => (
                <code key={r} className="reason-chip">{r}</code>
              ))}
            </div>
          )}

          {matchState.data.subsets.length === 0 && (
            <p className="no-subsets">No feasible bill subsets found for this payment amount.</p>
          )}

          {matchState.data.subsets.map((subset, i) => {
            const fbKey = matchState.data.event_id ?? "";
            const fb = feedback[fbKey];
            return (
              <div key={i} className={`subset-card ${i === 0 ? "subset-best" : ""}`}>
                <div className="subset-header">
                  <span className="subset-rank">#{i + 1}</span>
                  <span className="subset-sum">{subset.sum_display}</span>
                  {subset.abs_residual_minor > 0 && (
                    <span className="residual">
                      Δ {fmtAmount(subset.abs_residual_minor)}
                    </span>
                  )}
                </div>
                <ul className="bill-list">
                  {subset.bill_ids.map((bid) => (
                    <li key={bid} className="bill-chip">{bid}</li>
                  ))}
                </ul>
                {i === 0 && matchState.data.event_id && fb === undefined && (
                  <div className="feedback-row">
                    <button
                      className="btn-accept"
                      onClick={() => onAccept(matchState.data.event_id!)}
                    >
                      ✓ Accept
                    </button>
                    <button
                      className="btn-reject"
                      onClick={() => onReject(matchState.data.event_id!)}
                    >
                      ✗ Reject
                    </button>
                  </div>
                )}
                {i === 0 && fb !== undefined && (
                  <div className={`feedback-done feedback-${fb}`}>
                    {fb === "accepted" ? "Accepted ✓" : "Rejected ✗"}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── main app ────────────────────────────────────────────────────────────────

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listTenants()
      .then((ts) => {
        dispatch({ type: "SET_TENANTS", tenants: ts });
        if (ts.length > 0) {
          dispatch({ type: "SELECT_TENANT", id: ts[0].tenant_id });
        }
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

  const handleRunMatch = (payment: Payment) => {
    if (!state.selectedTenant) return;
    dispatch({ type: "MATCH_LOADING", paymentId: payment.payment_id });
    api
      .runMatch(state.selectedTenant, payment.payment_id)
      .then((r) => dispatch({ type: "MATCH_RESULT", paymentId: payment.payment_id, data: r }))
      .catch((e) =>
        dispatch({ type: "MATCH_ERROR", paymentId: payment.payment_id, message: String(e) })
      );
  };

  const handleAccept = (paymentId: string, eventId: string) => {
    api
      .recordOutcome(eventId, "accepted_as_is")
      .then(() => dispatch({ type: "FEEDBACK", eventId, status: "accepted" }))
      .catch((e) => setApiError(String(e)));
  };

  const handleReject = (paymentId: string, eventId: string) => {
    api
      .recordOutcome(eventId, "rejected")
      .then(() => dispatch({ type: "FEEDBACK", eventId, status: "rejected" }))
      .catch((e) => setApiError(String(e)));
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
          <button onClick={() => setApiError(null)}>✕</button>
        </div>
      )}

      <main className="payment-list">
        {state.payments.length === 0 && !apiError && (
          <p className="empty-state">No payments found. Run <code>bulk-match seed</code> to load demo data.</p>
        )}
        {state.payments.map((payment) => (
          <MatchPanel
            key={payment.payment_id}
            payment={payment}
            matchState={state.match[payment.payment_id] ?? { kind: "idle" }}
            feedback={state.feedback}
            onRunMatch={() => handleRunMatch(payment)}
            onAccept={(eid) => handleAccept(payment.payment_id, eid)}
            onReject={(eid) => handleReject(payment.payment_id, eid)}
          />
        ))}
      </main>
    </div>
  );
}
