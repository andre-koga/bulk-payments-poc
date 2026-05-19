const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface Payment {
  payment_id: string;
  tenant_id: string;
  amount_minor: number;
  amount_display: string;
  payment_date: string;
  counterparty_bank_name: string;
  description: string;
  source_currency: string | null;
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
}

export interface Tenant {
  tenant_id: string;
  ledger_currency: string;
  amount_tolerance_minor: number;
  date_window_days: number;
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
    userId?: string
  ) =>
    apiFetch<{ event_id: string; outcome: string }>(`/match-events/${eventId}/outcome`, {
      method: "POST",
      body: JSON.stringify({
        outcome,
        corrected_bill_ids: correctedBillIds ?? null,
        user_id: userId ?? null,
      }),
    }),
};
