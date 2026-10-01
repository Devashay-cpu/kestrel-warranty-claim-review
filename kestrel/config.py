"""Single source of truth for Phase 1 constants (all assumptions are explicit here)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"

# --- business rules (ops-policy v4.1) ---
REGIME_DATE = pd.Timestamp("2026-05-01")   # s5: claims < Rs 2,000 auto-approved from this date
SMALL_CLAIM_INR = 2000
CAPACITY_PER_MONTH = 40                    # investigation desk (s5, Finance email)
CLAIMS_PER_MONTH = 750                     # ~ test volume per month
GOODWILL_INR = 380                         # s4: held genuine claim
CONTACT_INR = 260                          # s4: blended service contact (assumption: investigation contact cost)

# --- modelling assumptions (documented in docs/PHASE1_DECISIONS.md) ---
LAG_DAYS = 14          # investigation lag: a label is "known" LAG_DAYS after submission
WINDOW_DAYS = 90       # recent-window partner history
EB_STRENGTH = 20       # empirical-Bayes pseudo-count for partner fraud rate
GLOBAL_PRIOR = 0.01    # neutral prior used only to stabilise the global rate at the very start
NEW_PARTNER_DAYS = 365

# --- untrusted free text: only these values are ever interpreted ---
ALLOWED_DESCRIPTIONS = [
    "remote not working", "loud noise while running", "power button not working",
    "filter indicator stuck", "motor not running", "water leaking", "not charging",
    "blade jammed", "tripping mcb", "burning smell", "unit not heating", "display not working",
]
ALLOWED_NOTES = [
    "Customer has bill, serial verified", "PCB replaced under warranty", "Photos match fault, approved",
    "Heating element open circuit", "Minor fault, part swapped", "Motor winding failure confirmed",
    "Unit inspected, fault confirmed",
]
