Details

Phase: Foundations · Estimated hours: 3

Why it matters: In this week's session Haris walked the whole river your data travels — Search Console and Analytics landing in BigQuery, the daily sync, the warehouse tables, and the safe release on Hugging Face. Your job now is small on purpose: describe YOUR lane's slice of that river, prove three facts about it with real queries, then build your first five features and spring the one trap everyone falls into. One notebook, and you're ready for the modeling weeks.

Your job: Four small things, in order. One — the contract, in plain words (5 answers): what one row means for your lane, which table(s) you'll use, which time window, what you'd predict or rank (label or proxy), and one thing you deliberately exclude. Two — prove three facts with three small queries on a mid-panel month (for example month=2026-03): the grain (one row really is what you said), your slice’s row count and date span, and availability — filter with IS TRUE and show how many rows survive. Three — five features, max: build a small feature frame for your lane from that same month, and give every feature one line: "knowable at the decision moment because…". Four — the trap: add ONE label-derived column on purpose, watch your quick score jump toward perfect, then delete it and keep the honest number — the leakage lesson from notebook 02, performed on real warehouse data by you.

Deliverable: your repo URL — with work/notebooks/w03_data_contract.ipynb executed and committed.

What done looks like: five plain-words contract answers; exactly three verification queries with their outputs visible (availability checked with IS TRUE); a five-feature frame with an "available when?" line per feature; the deliberate-leak experiment shown and removed; one named limitation of your slice.

Where this lives: in your repo, under work/notebooks/ — commit it there, then submit on this card. Done.

Your skeleton is ready: open work/notebooks/w03_data_contract.ipynb — contract answers in 1)–2), your three queries and the five-features + trap in 3), the limitation in 4), then 5) Self-check. Want to go deeper (optional)? The sibling skeleton w03_feature_leakage_check.ipynb is the full-depth version — the capstone rewards it, nothing requires it now.

Working with an AI assistant? Tell it to read skills/README.md in your repo, then load writing-data-contracts + flyrank/flyrank-data for this task.

Data access (2 minutes): on https://huggingface.co/datasets/FlyRank/internship-warehouse (opens in a new tab) press "request access" (instant approval), then create a READ token in your Hugging Face settings (pick the plain Read type — a fine-grained token can pass whoami and still 403 on the gated files unless its gated-repositories permission is ticked). In Colab, store it as a Secret named HF_TOKEN (the key panel) — never paste a token into a cell; your repo is public.

One warning before you iterate: the _sample table is not a random sample — it is exactly the final month (June 2026). Use it to test query mechanics, never to develop label logic: the last month is the natural outcome window of any past→future label. Iterate on a mid-panel month instead (for example month=2026-03) and treat the final month as a sealed test month.