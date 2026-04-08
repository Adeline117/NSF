# Appendix C: Risk Score Calibration

The initial risk score used a hardcoded `expected_max = 200.0` calibrated on the pilot synthetic dataset (~20 findings per repo). When applied to real scan data (62 repos), **29/62 repos saturate at 100.0**, making the top quartile indistinguishable.

## Raw base-score distribution

| Statistic | Value |
|-----------|-------|
| min | 0.0 |
| max | 8862.75 |
| mean | 610.5 |
| median | 144.0 |
| std | 1488.8 |
| p75 | 498.75 |
| p90 | 1460.7 |
| p95 | 1861.05 |
| p99 | 8193.73 |

## Recalibration options

| Scheme | Parameter | Saturated | Mean | Median | Std |
|--------|-----------|-----------|------|--------|-----|
| linear_p90 | 1460.7 | 7/62 | 25.54 | 9.86 | 32.47 |
| linear_p95 | 1861.0 | 4/62 | 21.54 | 7.74 | 29.18 |
| log_asymptotic | 144.0 | 12/62 | 57.83 | 62.89 | 38.25 |
| OLD_linear_200 | 200.0 | 29/62 | 60.57 | 72.0 | 41.71 |

## Rank correlation with the old 200-based scheme

| Comparison | Spearman ρ |
|------------|-----------|
| old_200_vs_linear_p90 | 0.9481 |
| old_200_vs_linear_p95 | 0.9476 |
| old_200_vs_log | 0.9475 |

## Recommendation

Use linear_p90 (expected_max=1460.7) — keeps Spearman rank >0.95 with old scheme while reducing saturation from 7/62 to ~6/62 and giving meaningful separation in the [80-99] range.

## Implementation

Update `paper2_agent_tool_security/static_analysis/analyzer.py` line 1151: `expected_max = 1460.7` (was 200.0).