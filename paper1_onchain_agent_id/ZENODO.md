# Getting a DOI via Zenodo

This directory is ready to archive on [Zenodo](https://zenodo.org) for a
permanent DOI. Two routes:

-----------------------------------------------------------------

## Option A — GitHub integration (recommended)

1. Log in to Zenodo with your GitHub account.
2. Visit https://zenodo.org/account/settings/github/ and flip the
   toggle for `adelinewen/NSF` to **On**.
3. In the GitHub repo, draft a new release:
   ```
   Tag:    p1-benchmark-v0.1.0
   Title:  OnChainAgentID benchmark v0.1.0
   ```
   In the release notes, paste the abstract from `.zenodo.json` and
   link to `paper1_onchain_agent_id/benchmark/README.md`.
4. Publishing the release triggers Zenodo to archive the tagged commit
   and mint a DOI within ~2 minutes.
5. Update `CITATION.cff` with the DOI it returns (`doi:` field at the
   top).

## Option B — Manual upload (if you don't want the GitHub webhook)

1. `git archive --format=zip --output p1-benchmark-v0.1.0.zip HEAD paper1_onchain_agent_id/`
2. On Zenodo → New Upload, drag the zip in and paste metadata from
   `.zenodo.json`.
3. Click Publish. Zenodo returns a DOI.
4. Add the DOI to `CITATION.cff` and `README.md`.

-----------------------------------------------------------------

## What gets archived

The archive should include:

```
paper1_onchain_agent_id/
├── benchmark/                       # README, datasheet, LICENSE, splits
├── onchain_audit/                   # Python package
├── notebooks/tutorial.ipynb         # 10-minute walkthrough
├── data/features_provenance_v4.parquet
├── data/labels_provenance_v4.json
├── experiments/                     # reproducer scripts
├── setup.py
├── CITATION.cff
└── .zenodo.json
```

Do **not** archive `data/raw/` (4,854 parquet files, ~2 GB). Users can
regenerate it via `experiments/mine_addresses_v4_1000plus.py`. See
`benchmark/README.md §5`.

-----------------------------------------------------------------

## Post-publication

Once Zenodo returns a DOI, update the following in-tree references:

- `CITATION.cff` — add a top-level `doi:` field.
- `benchmark/README.md` §6 — replace the GitHub URL in the bibtex with
  the DOI URL.
- `.zenodo.json` — add the DOI to `related_identifiers` as a
  self-reference.

Every new tagged GitHub release will mint a new version DOI under the
same concept DOI, so you can keep citing "the benchmark" stably.
