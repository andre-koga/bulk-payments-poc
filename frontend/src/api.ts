const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface PaymentSettlement {
  bill_ids: string[];
  allocated_sum_minor: number;
  remaining_minor: number;
  is_fully_allocated: boolean;
}

export interface Payment {
  payment_id: string;
  tenant_id: string;
  amount_minor: number;
  amount_display: string;
  payment_date: string;
  counterparty_bank_name: string;
  description: string;
  source_currency: string | null;
  /** Present when bills are already allocated to this payment (persists across reload). */
  settlement: PaymentSettlement | null;
}

export interface Bill {
  bill_id: string;
  vendor_id: string;
  vendor_name: string;
  open_amount_minor: number;
  amount_display: string;
  currency: string;
  open_date: string;
  due_date: string | null;
  matched_payment_id: string | null;
}

export interface SubsetResult {
  bill_ids: string[];
  sum_minor: number;
  abs_residual_minor: number;
  sum_display: string;
}

export interface MatchResult {
  payment_id: string;
  tenant_id: string;
  decision: "auto_applied" | "suggested" | "no_candidates" | "rejected_by_user" | "manual";
  unique_best: boolean;
  competing_subset_count: number;
  reason_codes: string[];
  subsets: SubsetResult[];
  ranker_score: number | null;
  calibrated_accept_prob: number | null;
  event_id: string | null;
  allocated_sum_minor: number;
  remaining_minor: number | null;
  is_fully_allocated: boolean;
}

export interface Tenant {
  tenant_id: string;
  ledger_currency: string;
  amount_tolerance_minor: number;
  date_window_days: number;
}

export interface AgentResolution {
  action: "propose_match" | "no_match" | "need_more_info";
  bill_ids: string[];
  confidence: number;
  reasoning: string;
  model_id: string;
  langsmith_run_id: string | null;
  langsmith_trace_url: string | null;
}

export interface MatchWithAgentResult {
  payment_id: string;
  tenant_id: string;
  event_id: string;
  resolution_id: string | null;
  rules: MatchResult;
  agent: AgentResolution | null;
}

export interface BatchAutoMatchResponse {
  tenant_id: string;
  summary: {
    processed: number;
    auto_matched: number;
    needs_review: number;
    no_match: number;
    skipped_fully_allocated: number;
  };
  auto_matched: Array<{
    payment_id: string;
    status: string;
    decision: string;
    bill_ids: string[];
  }>;
  needs_review: MatchResult[];
  no_match: Array<{ payment_id: string; status: string; decision: string }>;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listTenants: () => apiFetch<Tenant[]>("/tenants"),

  listPayments: (tenantId: string) =>
    apiFetch<Payment[]>(`/tenants/${tenantId}/payments`),

  listBills: (tenantId: string, unmatchedOnly = false) =>
    apiFetch<Bill[]>(`/tenants/${tenantId}/bills?unmatched_only=${unmatchedOnly}`),

  runMatch: (tenantId: string, paymentId: string) =>
    apiFetch<MatchResult>(`/tenants/${tenantId}/payments/${paymentId}/match`, {
      method: "POST",
    }),

  recordOutcome: (
    eventId: string,
    outcome: string,
    correctedBillIds?: string[],
    options?: { userId?: string; accountantReasoning?: string }
  ) =>
    apiFetch<{ event_id: string; outcome: string }>(`/match-events/${eventId}/outcome`, {
      method: "POST",
      body: JSON.stringify({
        outcome,
        corrected_bill_ids: correctedBillIds ?? null,
        user_id: options?.userId ?? null,
        accountant_reasoning: options?.accountantReasoning?.trim() || null,
      }),
    }),

  autoMatchAll: (tenantId: string) =>
    apiFetch<BatchAutoMatchResponse>(`/tenants/${tenantId}/auto-match`, {
      method: "POST",
    }),

  unmatchPayment: (tenantId: string, paymentId: string) =>
    apiFetch<{ payment_id: string; freed_bill_ids: string[]; event_id: string | null }>(
      `/tenants/${tenantId}/payments/${paymentId}/unmatch`,
      { method: "POST" }
    ),

  runAgentResolve: (tenantId: string, paymentId: string, model = "gpt-4o-mini") =>
    apiFetch<MatchWithAgentResult>(
      `/tenants/${tenantId}/payments/${paymentId}/agent-resolve`,
      {
        method: "POST",
        body: JSON.stringify({ model }),
      }
    ),

  latestAgentResolution: (tenantId: string, paymentId: string) =>
    apiFetch<AgentResolution | null>(
      `/tenants/${tenantId}/payments/${paymentId}/agent-resolution/latest`
    ),
};
