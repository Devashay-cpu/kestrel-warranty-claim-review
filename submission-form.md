Ques1 What did you build?
Ans I built a warranty-claim fraud ranking system that scores every test claim by estimated fraud risk and gives the investigation team reasons for reviewing a claim. The intended operating point is up to 40 reviews per month, with fewer reviews when the expected value is negative. In the central validation scenario, 120 reviews over three months could stop about Rs 51,377 of fraud with about Rs 32,300 goodwill cost, for an estimated net benefit of about Rs 19,077.

Ques2 Expected score:
Ans I expect a PR-AUC of about 0.28 on the hidden outcomes, with a plausible range of 0.17–0.52. I estimated this using time-based validation with June scenarios under different levels of history staleness. I am not claiming accuracy because the fraud rate is low and predicting no fraud already gives roughly 97–99% accuracy.

Ques3 How do you know it works?
Ans 43 automated Phase 4 tests pass, with 1 test skipped because the private test_unlabelled.csv is intentionally not included in the repository. Phase 2 and Phase 3 validation used time-based hold-outs and partner-cluster confidence intervals. The main failure mode is fraud appearing at outlets with no previous fraud history; stale outlet history is another major risk.


Ques4 Did you change/push back?
Ans Yes. I pushed back on using accuracy as the main KPI. With a roughly 1–3% fraud rate, a model predicting no fraud can exceed 97% accuracy while being useless for fraud detection. I therefore used PR-AUC, top-K precision, and operational economic value for model selection and deployment decisions.

Ques5 What is wrong with what you are handing us / the data?
Ans The labelled dataset contains only 141 fraud cases, including 36 after the May 2026 rule change, so validation uncertainty is high. Fraud is concentrated in a small number of outlets, and partner history becomes stale quickly. The hidden test outcomes are unavailable, so the expected PR-AUC is necessarily an estimate. Undecided claims were excluded, and four claims containing injected instructions in their descriptions were treated as untrusted data.

Ques 6What deliberately left out?
Ans I left out claim IDs, raw text/free-text descriptions, source fields and partner IDs as direct model features. Serial-number features and outlet age were also rejected during development because they did not provide sufficiently reliable evidence. I avoided free text because it was noisy and included injected instructions.

Ques7 Anything you built/found that nobody asked for?
Ans I added plain-language reasons to each prediction, economic break-even logic for deciding whether a claim should actually be investigated, stale-history stress tests, and monitoring recommendations for deployment.

Ques8 What did you use AI for?
Ans I used AI coding assistants and chat models mainly for debugging, implementation review, test interpretation, and exploring alternative approaches. They helped identify errors and speed up repetitive coding. I validated the resulting code myself through automated tests and validation runs. I discarded approaches that did not improve the evidence or were too complex for the available data. No paid API calls were used by the deployed service.

Then Google Drive recording link later.

Ques9 Someone picks this up Monday — three things?
Ans Three things for Monday
1. Refresh the model monthly with newly investigated claims and monitor precision of the reviewed queue.
2. Pay particular attention to new/unseen outlets and stale partner history because these are the main failure modes.
3. Never auto-reject a claim from the model score; use the score to prioritise investigation and apply the economic break-even rule.

Ques10Honest hours spent.** One number.*
Ans 15

Ques11 GitHub Repo Link
Ans https://github.com/Devashay-cpu/kestrel-warranty-claim-review

Ques12 Prediction  Cost
Ans One prediction costs approximately Rs 0 in paid API calls. The service runs locally using the trained model and preprocessing bundle. At approximately 750 claims per month, external model/API cost is 750 × Rs 0 = Rs 0 per month, excluding Kestrel's own compute and investigation/employee costs.