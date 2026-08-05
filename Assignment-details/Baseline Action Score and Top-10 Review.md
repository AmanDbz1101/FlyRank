Details

Phase: Build · Estimated hours: 3

Why it matters: This week's session showed you where FlyRank's flags come from — hand-written rules, honest thresholds, and the reasoning behind them. Now you do the same thing once, on your lane: check that the signals your rule leans on are real, encode the rule, and read your own top ten with a skeptic's eye. This baseline is what your Week-5 model must beat. And lanes lock this week — confirm or switch yours by the end.

Your job: Three small things in one notebook. One — check two signals first (one bucket table each, with n printed): pick two signals your rule idea leans on, and at least one must be a signal behind a real FlyRank flag from the session (staleness behind the refresh flags, CTR-vs-position behind the CTR-fix logic, volume behind quick-win). Give each a one-word verdict: CONFIRMED, OPPOSITE, MIXED, or FALSE — a clearly-explained negative is a win, and it just saved your rule. Two — encode ONE rule the way the session built one live: a score, ONE reason code, an action label; write the ranked queue to work/outputs/baseline_action_score.csv from the notebook. Three — the top-10 review: for each of your top ten, one line each — the action, why it's there, and what would make it wrong.

Deliverable: your repo URL — with work/notebooks/w04_baseline_score.ipynb executed and committed (it writes the CSV).

What done looks like: two signal verdicts with visible bucket tables and n (at least one flag-linked); one rule with a score, a reason code, and an action label; a ranked queue written from the notebook; ten reviewed rows with "what would make it wrong" for each; no future-window or label-derived inputs.

Where this lives: in your repo, under work/notebooks/ — commit it there, then submit on this card. Done.

Your skeleton is ready: open work/notebooks/w04_baseline_score.ipynb — your two signal checks live in 1) with your rule’s reasoning, the queue in 2), your top-10 in 3), weak picks in 4), then 5) Self-check. Want to go deeper (optional)? The sibling skeleton w04_signal_audit.ipynb is the full signal-audit version, and a top-20 review is always welcome — the capstone rewards both, nothing requires them.

Working with an AI assistant? Tell it to read skills/README.md in your repo, then load building-baselines + flyrank/flyrank-data for this task.

About the CSV: the notebook writes work/outputs/baseline_action_score.csv, and that file stays out of git by design — the CI leak-guard blocks data files, and your notebook regenerates it on every run. What IS worth committing: your metrics JSONs (work/outputs/*.json — your run’s receipts) and any figure you reuse.