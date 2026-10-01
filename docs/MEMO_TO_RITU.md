# Memo: warranty fraud checks

**To:** Ritu Deshpande, Head of D2C Operations  **From:** Kabir Nanda  **Date:** 1 Oct 2026

**What we built.** A ranking of every warranty claim by how likely it is to be fraud, using the past 15 months of investigated claims. It does not approve or reject anything. It tells your investigation desk which 40 claims to look at first each month, and why (for example: "outlet has 4 confirmed frauds in 25 claims", "small claim approved without inspection", "customer has 2 earlier claims").

**The number to watch is not accuracy.** If we flagged nothing at all we would still be 97-99% "accurate", because only 1 to 3 claims in 100 are fraud. The board's 97% bar is passed by doing nothing, so it says nothing about whether this works. Please put two numbers to the board instead: **rupees of fraud stopped per claim checked**, and **how many of the 40 checked claims were real fraud**.

**What it could be worth** (July to September, about 2,250 claims, 40 checks a month, Rs 380 goodwill for each genuine customer we hold). We cannot know the real result until investigations finish, so here is the range from our testing:
- Likely case: about 1 in 3 checked claims is fraud (35 of 120), Rs 51,000 of fraud stopped, Rs 32,000 goodwill spent, **about Rs 19,000 net, roughly Rs 6,400 a month**.
- Good case: Rs 35,000 net. Bad case, if fraud shifts to outlets we have no history on: a loss of about Rs 42,000, because almost every check hits a genuine customer.
- If each investigation also costs the Rs 260 service-contact cost, the likely case turns slightly negative (about minus Rs 12,000). The saving is real but thin, so the desk should check fewer claims, not more, when the model is unsure.

**Your hunch about new partners.** The data says it is a few outlets, not new partners as a group. Seven outlets account for 88 of the 120 claims we would check (SP3118, SP3286, SP3129, SP3160, SP3318, SP3319, SP3232). Most new partners show no fraud. Meenal's view holds.

**Next week**
1. Start the 40-a-month checks from the attached list; investigate, then record fraud / not fraud, quickly. Those results are what keep the model current.
2. Talk to or audit the seven outlets. Consider requiring inspection of their small claims, since the May change let small claims through unchecked.
3. After the first month, count how many of the 40 were fraud. If under about 1 in 5, pause and call us.
4. Ask Tanmay for investigation outcomes every month and the open cases, so we can refresh.

**Please do not**
- Reject or delay a customer just because the score is high. It is a reason to look, not a verdict.
- Label all new partners as risky, or rely on the accuracy figure.
- Fill all 40 slots with low-value claims: below roughly Rs 800 a check costs more than it saves unless we are very sure.

**Biggest risk.** Fraud moves faster than the history we learn from. If a new outlet starts the same pattern, the model will not see it until a few cases are investigated. Monthly refresh and the monthly precision count are the safeguard.
