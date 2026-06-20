"""
Pattern 9: Human-in-the-Loop
==============================
> Google Cloud Reference: https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system#human-in-the-loop-pattern

The Human-in-the-Loop pattern integrates mandatory human review checkpoints
into an agent's workflow. At predefined steps, the agent PAUSES and waits
for a human to approve, correct, or provide input before proceeding.

Use for:
- High-stakes decisions (large financial transactions, medical data, legal docs)
- Safety-critical operations
- Subjective approvals (creative content, policy decisions)
- Compliance requirements (GDPR, HIPAA, SOX)

Architecture:
    Agent processes task
          │
          ▼
    [Checkpoint] ──► Agent pauses ──► Human reviews ──► Agent continues / stops
          │
    (if approved)
          │
          ▼
    Agent completes task

Key implementation: Agno's `confirm_action` hook and `user_control` parameter.

Run: python 4_human_in_loop.py
"""

import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
for var in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']:
    if os.getenv(var):
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from agno.agent import Agent
from agno.models.openai import OpenAIChat


class RunResponse:
    """Minimal result container — installed agno version dropped the old
    class-based Workflow API, so these demos drive Agents directly."""
    def __init__(self, content: str):
        self.content = content


# ─────────────────────────────────────────────────────────────
# Helper: Human approval checkpoint
# ─────────────────────────────────────────────────────────────

def request_human_approval(
    action: str,
    details: dict,
    context: str = "",
    risk_level: str = "MEDIUM",
) -> bool:
    """
    Pause execution and request human approval.
    In production, this would call a notification service (Slack, email, etc.)
    and wait for async human input.
    """
    print(f"\n{'⚠️ ' * 20}")
    print(f"🛑 HUMAN APPROVAL REQUIRED [{risk_level} RISK]")
    print(f"{'⚠️ ' * 20}")
    print(f"\nAction:  {action}")
    print(f"Context: {context}")
    print(f"\nDetails:")
    for key, value in details.items():
        print(f"  {key}: {value}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        decision = input(f"\n[{risk_level}] Approve? (y=yes / n=no / m=modify): ").strip().lower()
        if decision in ['y', 'yes']:
            print("✅ Approved by human reviewer.")
            return True
        elif decision in ['n', 'no']:
            print("❌ Rejected by human reviewer.")
            return False
        elif decision in ['m', 'modify']:
            modification = input("Enter modification note: ").strip()
            print(f"📝 Modification noted: {modification}")
            return True  # Approved with modification note
        else:
            print("Please enter y, n, or m.")


# ─────────────────────────────────────────────────────────────
# Example 1: Patient Data Anonymization with HIPAA Compliance Check
# ─────────────────────────────────────────────────────────────

class AnonymizationReport(BaseModel):
    pii_found: list[str] = Field(..., description="List of PII elements found")
    anonymized_text: str = Field(..., description="Text with PII redacted")
    redaction_count: int = Field(..., description="Number of PII elements redacted")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in completeness")
    requires_human_review: bool = Field(..., description="True if any ambiguous PII detected")
    review_reason: Optional[str] = Field(None, description="Why human review is needed")

model = OpenAIChat(id="google/gemini-2.5-flash")

class HIPAAAnonymizationWorkflow:
    """
    Human-in-the-Loop Pattern: AI anonymizes patient data, human validates before release.
    Mandatory human checkpoint before any data is released.
    """

    def __init__(self):
        self.pii_detector = Agent(
            name="PII Detector & Anonymizer",
            model=model,
            output_schema=AnonymizationReport,
            structured_outputs=True,
            instructions=[
                "You are a HIPAA compliance AI.",
                "Identify ALL Protected Health Information (PHI): names, DOB, SSN, MRN, addresses, phone numbers, email, dates, device IDs.",
                "Replace each PHI element with [REDACTED-TYPE] (e.g., [REDACTED-NAME], [REDACTED-DOB]).",
                "Set requires_human_review=True if you find any ambiguous cases.",
                "Set confidence_score based on completeness of anonymization.",
            ],
        )

    def run(self, patient_record: str, release_purpose: str) -> RunResponse:
        print("\n" + "="*65)
        print("HIPAA ANONYMIZATION WORKFLOW (Human-in-the-Loop)")
        print("="*65)

        # Step 1: AI anonymizes the data
        print("\n🤖 Step 1: AI analyzing and anonymizing patient data...")
        result = self.pii_detector.run(
            f"Anonymize this patient record for: {release_purpose}\n\n{patient_record}"
        )
        report: AnonymizationReport = result.content

        print(f"   PII elements found: {len(report.pii_found)}")
        print(f"   Redactions made: {report.redaction_count}")
        print(f"   Confidence: {report.confidence_score:.0%}")
        print(f"   Requires review: {report.requires_human_review}")

        # Step 2: MANDATORY human checkpoint
        print("\n📋 Step 2: Preparing human review packet...")
        approved = request_human_approval(
            action="RELEASE anonymized patient data for research",
            details={
                "Purpose": release_purpose,
                "PII Elements Found": ", ".join(report.pii_found[:5]) + ("..." if len(report.pii_found) > 5 else ""),
                "Redaction Count": report.redaction_count,
                "AI Confidence": f"{report.confidence_score:.0%}",
                "Flagged for Review": report.requires_human_review,
                "Review Reason": report.review_reason or "N/A",
            },
            context="Patient data release for medical research requires compliance officer approval.",
            risk_level="HIGH",
        )

        if not approved:
            return RunResponse(
                content="❌ Data release BLOCKED by compliance officer. Record has been flagged for manual review."
            )

        # Step 3: Human approved — release the data
        print("\n📤 Step 3: Releasing anonymized data (human-approved)...")
        timestamp = datetime.now().isoformat()
        audit_log = {
            "timestamp": timestamp,
            "action": "DATA_RELEASED",
            "approver": "Human Compliance Officer",
            "purpose": release_purpose,
            "redaction_count": report.redaction_count,
            "confidence": report.confidence_score,
        }

        return RunResponse(
            content=f"""## ✅ HIPAA-Compliant Data Release (Human-Approved)

**Audit Log:** {json.dumps(audit_log, indent=2)}

**Anonymized Record:**
{report.anonymized_text}
"""
        )


# ─────────────────────────────────────────────────────────────
# Example 2: Financial Transaction Approval
# ─────────────────────────────────────────────────────────────

class TransactionPlan(BaseModel):
    transaction_type: str
    amount_usd: float
    from_account: str
    to_account: str
    reason: str
    risk_assessment: str
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    recommended_action: str

class FinancialApprovalWorkflow:
    """
    Human-in-the-Loop: AI analyzes transaction, human approves before execution.
    Automatic approval for amounts < $1000, human required for >= $1000.
    """

    def __init__(self):
        self.transaction_analyzer = Agent(
            name="Transaction Analyzer",
            model=model,
            output_schema=TransactionPlan,
            structured_outputs=True,
            instructions=[
                "Analyze financial transactions for risk.",
                "Set risk_level: LOW (<$500), MEDIUM ($500-$5000), HIGH ($5000-$50000), CRITICAL (>$50000).",
                "Flag anything unusual: new recipient, unusual amount, weekend transfer, international.",
                "recommended_action: 'auto-approve' for LOW, 'human-review' for others.",
            ],
        )

    def run(self, transaction_request: str) -> RunResponse:
        print("\n" + "="*65)
        print("FINANCIAL TRANSACTION APPROVAL WORKFLOW")
        print("="*65)

        print("\n🤖 Step 1: AI analyzing transaction risk...")
        result = self.transaction_analyzer.run(transaction_request)
        plan: TransactionPlan = result.content

        print(f"\nTransaction: ${plan.amount_usd:,.2f} | Risk: {plan.risk_level}")
        print(f"From: {plan.from_account} → To: {plan.to_account}")
        print(f"AI Recommendation: {plan.recommended_action}")

        # Auto-approve low risk
        if plan.risk_level == "LOW" and plan.recommended_action == "auto-approve":
            print("\n✅ AUTO-APPROVED (Low risk, below threshold)")
            return RunResponse(
                content=f"✅ Transaction auto-approved: ${plan.amount_usd:,.2f} to {plan.to_account}"
            )

        # Human-in-the-loop for medium/high/critical
        approved = request_human_approval(
            action=f"Execute {plan.transaction_type} of ${plan.amount_usd:,.2f}",
            details={
                "Amount": f"${plan.amount_usd:,.2f}",
                "From": plan.from_account,
                "To": plan.to_account,
                "Reason": plan.reason,
                "Risk Assessment": plan.risk_assessment,
            },
            context="Large transactions require manager authorization per company policy.",
            risk_level=plan.risk_level,
        )

        if approved:
            return RunResponse(
                content=f"✅ Transaction APPROVED by authorized reviewer: ${plan.amount_usd:,.2f} to {plan.to_account}"
            )
        else:
            return RunResponse(
                content=f"❌ Transaction BLOCKED by reviewer: ${plan.amount_usd:,.2f} to {plan.to_account}"
            )


# ─────────────────────────────────────────────────────────────
# Run the demos
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🔐 " * 20)
    print("HUMAN-IN-THE-LOOP PATTERN DEMOS")
    print("🔐 " * 20)

    # Demo 1: HIPAA Data Release
    patient_record = """
    Patient: John Michael Smith, DOB: 03/15/1978, MRN: 7734521
    Address: 123 Oak Street, Boston, MA 02101
    Phone: (617) 555-0182, Email: jsmith78@gmail.com
    Diagnosis: Type 2 Diabetes (ICD-10: E11.9)
    Medications: Metformin 1000mg twice daily
    Lab Results: HbA1c 7.2% (2024-11-15), Glucose fasting 142 mg/dL
    Treating Physician: Dr. Sarah Chen, License #MA-45892
    """

    hipaa_workflow = HIPAAAnonymizationWorkflow()
    result1 = hipaa_workflow.run(
        patient_record=patient_record,
        release_purpose="Diabetes treatment outcomes research study"
    )
    print("\n" + result1.content)

    print("\n\n" + "💰 " * 20)

    # Demo 2: Financial Transaction
    fin_workflow = FinancialApprovalWorkflow()
    result2 = fin_workflow.run(
        "Transfer $15,000 from operations account (ACC-001) to new vendor PaymentsXYZ (ACC-999) "
        "for Q1 software licenses. Vendor was onboarded 3 days ago."
    )
    print("\n" + result2.content)

    print("\n✅ Human-in-the-Loop demos complete.")
    print("Key insight: Human oversight is a pattern, not just a feature — design it into the architecture.")
