"""Tool: confirm a Nano (XNO) payment by verifying its block off-chain.

Nano is verification == settlement: an exact-amount send receivable TO the
merchant's own nano_ address IS the settlement. No facilitator, no gas, no
token account, no relay. The merchant needs only its public address.

Audit invariant (payee binds): the proof of settlement lives on the
merchant's own chain, not on a hash the payer chose to show. So verification
does two independent reads of the public ledger:
  1. the payer-named block: its amount and destination must match exactly; and
  2. the merchant's account history: the same block must appear among the
     merchant's real incoming receive-blocks of that exact amount.
The payer's hash is a HINT, never the proof. Settlement is only "confirmed"
when the block the customer names is also a receive the merchant actually
holds. A payer can then re-derive the same link by reading the merchant's
public account, and neither side controls the other's view of the chain.
"""

from __future__ import annotations

import urllib.request
import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Public Nano RPC (no key required for read-only block_info/account_history).
# Swap for a configured node URL if the deployer prefers one.
RPC_URL = "https://rpc.nano.to"
# rpc.nano.to refuses requests with no User-Agent (HTTP 403), so send one.
_USER_AGENT = "langgraph-sales-agent-nano-leg/1.0"


def _rpc(method: str, params: dict):
    body = json.dumps({"action": method, **params}).encode()
    req = urllib.request.Request(
        RPC_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


@tool
def verify_nano_block(
    block_hash: str, expected_raw: int, config: RunnableConfig
) -> str:
    """Confirm a Nano payment against the merchant's receivable on-chain.

    Args:
        block_hash: The send/receive block hash the customer provided. This is
            a hint used to cross-check, never the proof on its own.
        expected_raw: The exact raw amount the invoice required.
    """
    merchant = config["configurable"]["merchant_nano_address"]
    try:
        info = _rpc("block_info", {"json_block": "true", "hash": block_hash})
    except Exception as exc:  # network / RPC failure is not a confirmation
        return f"could-not-verify: {exc}"

    if info.get("error"):
        return "not-yet: block not found (still pending or invalid hash)"

    # 1. The payer-named block must be an exact-amount receive to the merchant.
    amount_raw = int(info.get("amount", 0))
    if amount_raw != expected_raw:
        return (
            f"amount-mismatch: got {amount_raw} raw, invoice required "
            f"{expected_raw} raw. Do not release the order."
        )
    # A block's owning account is `block_account` on a modern node; some nodes
    # expose it only under contents.account. Read either, never assume `account`.
    block_owner = info.get("block_account") or (info.get("contents") or {}).get("account")
    if block_owner != merchant:
        return "wrong-destination: block was not received to the merchant address"

    # 2. The payee-bound proof: the exact-amount receive must ALSO appear in
    #    the merchant's own account history, independent of the payer's claim.
    #    An open block (an account's first receive) is the OLDEST entry, so a
    #    single fixed window can never see it — walk the history in pages.
    try:
        found = _history_has_receive(merchant, block_hash, expected_raw, info.get("source"))
    except Exception as exc:
        # History read failed but the block itself is sound: trust the
        # direct block_info only if history is temporarily unavailable.
        return (
            f"audit-incomplete: history unreadable ({exc}); "
            f"block hash {block_hash} amount OK"
        )
    if not found:
        return (
            "not-in-merchant-history: the named block is not one of the "
            "merchant's real receives of that exact amount. A payer cannot "
            "bind settlement with a hash the merchant never received."
        )
    return "confirmed"


def _history_has_receive(
    merchant: str, block_hash: str, expected_raw: int, source: str | None
) -> bool:
    """Walk the merchant's account history looking for a matching receive.

    Pages until the named hash is found (as a receive of the exact amount) or
    the history is exhausted. Stops after a bounded number of pages so a huge
    account cannot hang the tool; a receive older than that is not settlement
    for a fresh invoice anyway.
    """
    cursor = None
    for _ in range(20):  # up to 20 * 100 = 2000 blocks back
        params = {"account": merchant, "count": 100, "raw": "true"}
        if cursor:
            params["head"] = cursor
        page = _rpc("account_history", params)
        if not isinstance(page, dict) or page.get("error") or not page.get("history"):
            return False
        history = page["history"]
        for h in history:
            if h.get("type") != "receive":
                continue
            if str(h.get("amount", "")) != str(expected_raw):
                continue
            if h.get("hash") == block_hash or (source and h.get("hash") == source):
                return True
        # A short page means we reached the account's open block: stop.
        if len(history) < 100:
            return False
        cursor = history[-1].get("hash")
    return False