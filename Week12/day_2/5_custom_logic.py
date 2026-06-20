"""
Pattern 10: Custom Logic Pattern
==================================
> Google Cloud Reference: https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system#custom-logic-pattern

The Custom Logic Pattern gives you maximum flexibility — you orchestrate agents
with your own Python code: conditionals, loops, branching, business rules.

Use for:
- Workflows that mix multiple patterns (parallel + sequential + conditional)
- When no standard pattern fits your business logic
- When you need fine-grained control at specific steps

Example: Customer Refund System (from Google Cloud guide)
  1. Verify purchaser AND check refund eligibility (parallel)
  2. If eligible → process refund
  3. If NOT eligible → store credit path (sequential: calc credit → approve → issue)

Architecture (from GCP guide):
    User Query
         │
         ▼
    [Coordinator] ──► [Parallel Verifier] ──► check_refund_eligibility()
                            │                         │
                    [Purchaser Verifier]       (eligible?)
                    [Refund Eligibility]      /         \\
                                        YES             NO
                                         │               │
                                   [Refund        [Store Credit]──► [Process Credit]
                                   Processor]
                                         │               │
                                         └───────────────┘
                                                 │
                                         [Response Agent]

Run: python 5_custom_logic.py
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.models.openai import OpenAIChat
import asyncio


model = OpenAIChat(id="google/gemini-2.5-flash")


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────

class PurchaseVerification(BaseModel):
    verified: bool
    customer_name: str
    order_id: str
    purchase_date: str
    item: str
    amount: float
    reason: str

class RefundEligibility(BaseModel):
    eligible: bool
    reason: str
    days_since_purchase: int
    policy_applied: str

class RefundResult(BaseModel):
    success: bool
    refund_amount: float
    refund_method: str
    processing_days: int
    confirmation_number: str
    message: str

class StoreCreditResult(BaseModel):
    credit_amount: float
    credit_percentage_bonus: float
    expiry_days: int
    credit_code: str
    reason_denied_refund: str
    message: str

class CustomerResponse(BaseModel):
    resolution: str = Field(..., description="One of: refund, store_credit, denied")
    customer_message: str = Field(..., description="Friendly message to the customer")
    next_steps: list[str] = Field(..., description="What the customer should do next")
    escalation_needed: bool = Field(default=False)


# ─────────────────────────────────────────────────────────────
# Specialized Agents
# ─────────────────────────────────────────────────────────────

purchaser_verifier = Agent(
    name="Purchase Verifier",
    model=model,
    output_schema=PurchaseVerification,
    structured_outputs=True,
    instructions=[
        "Verify if the customer made the purchase they're claiming.",
        "Simulate a database lookup — most orders are valid unless the order ID looks fake.",
        "Set verified=False only for clearly invalid/non-existent order IDs.",
        "Populate all fields based on reasonable assumptions.",
    ],
)

refund_eligibility_checker = Agent(
    name="Refund Eligibility Checker",
    model=model,
    output_schema=RefundEligibility,
    structured_outputs=True,
    instructions=[
        "Check refund eligibility based on company policy.",
        "Policy: Within 30 days = eligible. 31-60 days = store credit only. >60 days = denied.",
        "Software/digital goods: 14-day return window.",
        "Damaged items: always eligible regardless of time.",
        "Simulate days_since_purchase based on purchase date context.",
    ],
)

refund_processor = Agent(
    name="Refund Processor",
    model=model,
    output_schema=RefundResult,
    structured_outputs=True,
    instructions=[
        "Process the refund and generate a confirmation.",
        "Generate a realistic confirmation_number (e.g., REF-XXXXXX).",
        "Standard refunds: 5-7 days. Express: 2-3 days (for amounts >$200).",
        "Refund to original payment method by default.",
    ],
)

store_credit_agent = Agent(
    name="Store Credit Agent",
    model=model,
    output_schema=StoreCreditResult,
    structured_outputs=True,
    instructions=[
        "Calculate store credit as an alternative to a refund.",
        "Offer 10% bonus credit over the refund amount as goodwill.",
        "Credit expires in 365 days.",
        "Generate a credit code in format: CREDIT-XXXXXX.",
        "Clearly explain why a direct refund was not possible.",
    ],
)

response_composer = Agent(
    name="Customer Response Composer",
    model=model,
    output_schema=CustomerResponse,
    structured_outputs=True,
    instructions=[
        "Compose a warm, empathetic response to the customer.",
        "Clearly explain what was decided and why.",
        "Provide clear next steps.",
        "For denied claims, acknowledge frustration and explain escalation options.",
        "Never be robotic — sound human and caring.",
    ],
)


# ─────────────────────────────────────────────────────────────
# Custom Logic Orchestrator
# ─────────────────────────────────────────────────────────────

async def arun_structured(agent: Agent, prompt: str, retries: int = 2):
    """gemini-2.5-flash occasionally returns truncated/malformed JSON, which
    agno can't parse into the output_schema — it falls back to a raw string.
    Retry a couple of times before giving up."""
    result = None
    for attempt in range(retries + 1):
        result = await agent.arun(prompt)
        if not isinstance(result.content, str):
            return result
        print(f"   ⚠️  {agent.name}: malformed structured output, retrying ({attempt + 1}/{retries})...")
    raise RuntimeError(
        f"Agent '{agent.name}' failed to return structured output after {retries + 1} attempts. "
        f"Last raw response: {result.content!r}"
    )


async def process_refund_request(customer_query: str) -> str:
    """
    Custom logic orchestration — mixes parallel execution, conditional branching,
    and sequential processing. This is the 'Custom Logic Pattern'.
    """
    print("\n" + "="*65)
    print("CUSTOM LOGIC PATTERN: Customer Refund System")
    print("="*65)
    print(f"\nCustomer Request: {customer_query[:80]}...")

    # ── Step 1: PARALLEL verification ──────────────────────────
    print("\n🔄 Step 1: Running parallel verification (Purchaser + Eligibility)...")
    verify_task = arun_structured(
        purchaser_verifier, f"Verify this customer's purchase claim: {customer_query}"
    )
    eligibility_task = arun_structured(
        refund_eligibility_checker, f"Check refund eligibility for: {customer_query}"
    )

    verify_result, eligibility_result = await asyncio.gather(
        verify_task, eligibility_task
    )

    verification: PurchaseVerification = verify_result.content
    eligibility: RefundEligibility = eligibility_result.content

    print(f"   Purchase verified: {verification.verified} | Customer: {verification.customer_name}")
    print(f"   Refund eligible: {eligibility.eligible} | Days since purchase: {eligibility.days_since_purchase}")

    # ── Step 2: CUSTOM CONDITIONAL BRANCH ──────────────────────
    resolution_data = {}

    if not verification.verified:
        # Branch A: Purchase not verified — deny
        print("\n❌ Branch A: Purchase NOT verified → Deny claim")
        resolution_type = "denied"
        resolution_data = {"reason": "Purchase could not be verified in our system."}

    elif eligibility.eligible:
        # Branch B: Eligible → Process refund
        print("\n✅ Branch B: ELIGIBLE → Processing refund...")
        refund_result_raw = await arun_structured(
            refund_processor,
            f"Process refund for: Customer={verification.customer_name}, "
            f"Amount=${verification.amount}, Item={verification.item}"
        )
        refund: RefundResult = refund_result_raw.content
        resolution_type = "refund"
        resolution_data = {
            "amount": refund.refund_amount,
            "method": refund.refund_method,
            "days": refund.processing_days,
            "confirmation": refund.confirmation_number,
        }
        print(f"   Refund: ${refund.refund_amount} via {refund.refund_method} ({refund.processing_days} days)")
        print(f"   Confirmation: {refund.confirmation_number}")

    else:
        # Branch C: Not eligible → Store credit path (sequential)
        print("\n💳 Branch C: NOT eligible → Store credit path (sequential)...")

        # Step C1: Calculate store credit
        print("   C1: Calculating store credit...")
        credit_result_raw = await arun_structured(
            store_credit_agent,
            f"Calculate store credit for: Customer={verification.customer_name}, "
            f"Purchase amount=${verification.amount}, "
            f"Reason ineligible: {eligibility.reason}, "
            f"Policy applied: {eligibility.policy_applied}"
        )
        credit: StoreCreditResult = credit_result_raw.content
        resolution_type = "store_credit"
        resolution_data = {
            "credit_amount": credit.credit_amount,
            "bonus": credit.credit_percentage_bonus,
            "code": credit.credit_code,
            "expiry_days": credit.expiry_days,
        }
        print(f"   Store credit: ${credit.credit_amount:.2f} (includes {credit.credit_percentage_bonus}% bonus)")
        print(f"   Credit code: {credit.credit_code}")

    # ── Step 3: Compose customer response ──────────────────────
    print("\n📝 Step 3: Composing customer response...")
    response_context = json.dumps({
        "resolution_type": resolution_type,
        "resolution_details": resolution_data,
        "customer_name": verification.customer_name,
        "item": verification.item,
        "original_query": customer_query,
    })

    response_raw = await arun_structured(
        response_composer,
        f"Compose a customer response for this resolution:\n{response_context}"
    )
    response: CustomerResponse = response_raw.content

    # ── Final Output ────────────────────────────────────────────
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║              REFUND REQUEST — RESOLUTION                     ║
╠══════════════════════════════════════════════════════════════╣
║  Resolution: {response.resolution.upper():<50}║
╚══════════════════════════════════════════════════════════════╝

📧 Message to Customer:
{response.customer_message}

📋 Next Steps:
{chr(10).join(f'  {i+1}. {step}' for i, step in enumerate(response.next_steps))}

{'⚠️  ESCALATION FLAG: This case needs human review.' if response.escalation_needed else ''}
"""
    return output


# ─────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────

async def main():
    print("🔧 CUSTOM LOGIC PATTERN DEMONSTRATION")
    print("=" * 65)
    print("This pattern mixes: Parallel verification + Conditional branching")
    print("+ Sequential processing — all orchestrated in custom Python code.")

    test_cases = [
        {
            "label": "ELIGIBLE REFUND (recent purchase)",
            "query": (
                "I bought a laptop (Order #LP-78234) 10 days ago for $1,299 and "
                "the screen is cracked. I'd like a full refund. "
                "My name is Sarah Johnson and I paid with Visa."
            ),
        },
        {
            "label": "NOT ELIGIBLE — STORE CREDIT PATH",
            "query": (
                "I purchased a Bluetooth speaker (Order #BT-55512) 45 days ago for $89. "
                "I changed my mind and don't need it anymore. Can I get a refund? "
                "Customer name: Mike Chen."
            ),
        },
        {
            "label": "VERIFICATION FAILURE",
            "query": (
                "I want a refund for order #FAKE-99999 for $500. "
                "I don't remember what I bought or when."
            ),
        },
    ]

    for tc in test_cases:
        print(f"\n\n{'🔹 ' * 25}")
        print(f"TEST: {tc['label']}")
        print('🔹 ' * 25)
        result = await process_refund_request(tc["query"])
        print(result)

    print("\n✅ Custom Logic Pattern demo complete.")
    print("Key takeaway: Custom logic = full Python control over multi-pattern orchestration.")


if __name__ == "__main__":
    asyncio.run(main())
