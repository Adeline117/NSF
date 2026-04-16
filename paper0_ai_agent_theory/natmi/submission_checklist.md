# Nature Machine Intelligence Submission Checklist

## Manuscript
- [x] Title page with title, author list, affiliations
- [x] Abstract (target 200 words; currently ~190 words)
- [x] Keywords provided
- [x] Main text: Introduction, Main Results, Mechanistic Insight, Cross-Domain Validation, Implications, Methods
- [x] All numbers cross-checked against experiments/*.json
- [x] References complete (main.bib, 46 entries)
- [x] Paper compiles cleanly with pdflatex + bibtex (20 pages)
- [ ] Replace "NSF Project Team" with real author names and affiliations
- [ ] Update "Author affiliations withheld for double-blind review" if NatMI is not double-blind
- [ ] Add ORCID identifiers for all authors
- [ ] Final word count check (NatMI Articles: ~3,000-5,000 words main text)

## Figures
- [x] Fig 1: MI vs accuracy (Fano ceiling curve + measured points) -- fig1_mi_vs_accuracy.pdf
- [x] Fig 2: Clustering saturation (silhouette vs k + HDBSCAN) -- fig2_cluster_saturation.pdf
- [x] Fig 3: Execution vs decision mechanism (paired bar chart) -- fig3_execution_vs_decision.pdf
- [x] Fig 4: Cross-chain ETH vs Polygon comparison -- fig4_cross_chain.pdf
- [x] All figures generated as vector PDF (300 dpi)
- [ ] Verify figure resolution meets NatMI requirements (typically 300 dpi minimum)
- [ ] Prepare separate high-resolution figure files if required by submission portal

## Tables
- [x] Table 1: MI, conditional entropy, Fano ceiling, measured accuracy for X_23/X_31/X_47

## Supplementary Materials (referenced but not yet written)
- [ ] Supplementary Note 1: Full taxonomy details and provenance sources per category
- [ ] Supplementary Note 2: Exhaustive feature definitions (47 features, 6 families)
- [ ] Supplementary Note 3: Pre-registration appendix (hyperparameters, class boundaries)
- [ ] Supplementary Table S1: Loose vs tight Fano bounds comparison
- [ ] Supplementary Table S2: Pre-registered primary and secondary analyses
- [ ] Supplementary Table S3: Per-feature mutual information values

## Data and Code Availability
- [ ] OSF archive created and DOI reserved
- [ ] Feature parquets deposited
- [ ] I(X;Y) estimator outputs deposited
- [ ] Classifier predictions deposited
- [ ] Docker image for end-to-end reproduction tested on blank Ubuntu 22.04 VM
- [ ] Fano-bound inversion module (~120 lines) released under MIT licence
- [ ] Code repository linked in OSF archive

## Cover Letter
- [x] Cover letter written (cover_letter.txt)
- [ ] Add suggested reviewers (3-5 names with expertise in info theory / AI governance / agent detection)
- [ ] Add conflict-of-interest declarations

## NatMI-Specific Requirements
- [ ] Check current NatMI formatting requirements (https://www.nature.com/natmachintell/for-authors)
- [ ] Verify reference format matches NatMI style (currently unsrt; NatMI may use Nature style)
- [ ] Methods section placement: NatMI may require Methods at end (currently at end -- OK)
- [ ] Data availability statement present (line ~930)
- [ ] Code availability statement present (line ~957)
- [ ] Ethics statement present (line ~937)
- [ ] Competing interests declaration needed
- [ ] Author contributions statement needed

## Pre-Submission Checks
- [ ] Run plagiarism/similarity check
- [ ] Verify all citations resolve (no undefined references)
- [ ] Check for orphaned/dangling cross-references
- [ ] Spell-check entire manuscript
- [ ] Have at least one co-author do a final read-through
- [ ] Confirm submission portal account is set up
