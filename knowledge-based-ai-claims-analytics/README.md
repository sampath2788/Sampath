# Data-Quality-Aware Knowledge-Based AI-Driven Claims Analytics

**Research status:** experimental Short Communication draft for *Knowledge-Based Systems* (Elsevier), **not** a published or accepted manuscript.

This study explores whether explicit evidence-sufficiency checks and three-valued knowledge rules can improve the safety of selective automated coverage-evidence review using synthetic insurance claims.

The code is research-only: it must not be used for benefits adjudication, real patient decisions, claims denial, fraud labeling, or production deployment. No company data, protected health information, or confidential insurer policies are included.

## Reproduction

```bash
python -m pip install -r requirements.txt
pytest -q test_claims_benchmark.py
python claims_benchmark.py --out results --target-risk 0.03
```

The controlled experiment uses nine random seeds and compares plain ML, symbolic rules, rules-plus-ML, selective ML, and a knowledge-based quality-aware approach. Metrics include automation coverage, selective error, false-negative decisions, and incorrect definitive decisions per 1,000. The experiments distinguish missing values from plausible but silently mutated values, which are a known limitation.

The **manuscript is not submission-ready** without independent benchmark evidence and further assessment of novelty and explanation fidelity. A low overall error count can arise from deferring more records; paired comparisons and matched automation coverage matter.

Results, figures, and tables are generated from the experiment script. The intended submission has exactly two figures and two editable tables. The official journal author guide limits Short Communications to 10 double-spaced pages including tables and figures. Do not reuse the illustrative synthetic results as real-world insurance estimates.

### Data and license

All records are simulated at runtime. No open-source license has yet been selected; please confirm license before redistribution. Do not upload patient or employer data to this repository.

### Code and documents

This repository is maintained as a private working area. A Word manuscript, graphical PDFs, and journal audit documents may be retained locally until reviewed for public disclosure. Do not add unpublished third-party reference PDFs or unrelated backups.
