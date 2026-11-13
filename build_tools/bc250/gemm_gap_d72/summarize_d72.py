#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D72 pure summarizer: attribute the in-model-vs-fixture GEMM gap.

Combines the three fixture experiments into a single attribution against the
pre-declared thresholds (MEANINGFUL_RATIO 1.5, DOMINANT_RATIO 2.5 — both above
D71's 0.96-1.05 noise band). The primary cause is the hypothesis with the
largest measured ratio at or above MEANINGFUL_RATIO; if none reaches it, the
gap is not reproduced by isolated fixtures and needs in-model investigation.
No I/O; fully unit-testable.
"""

from __future__ import annotations

MEANINGFUL_RATIO = 1.5
DOMINANT_RATIO = 2.5


def _ratio(alt_impl: dict, sustained: dict, locality: dict) -> dict:
    dom = alt_impl["dominant"]
    return {
        "alt_impl_backward_forward": dom["backward_forward_ratio"],
        "alt_impl_best_backward_forward": dom["best_backward_forward_ratio"],
        "sustained_degradation": sustained["degradation_ratio"],
        "locality_cold_hot": locality["cold_hot_ratio"],
        "locality_fresh_hot": locality["fresh_hot_ratio"],
    }


def build_summary(alt_impl: dict, sustained: dict, locality: dict) -> dict:
    for name, report in (
        ("alt-impl", alt_impl),
        ("sustained", sustained),
        ("locality", locality),
    ):
        if report.get("status") != "passed":
            return dict(status="failed", error=f"{name} status {report.get('status')}")
    ratios = _ratio(alt_impl, sustained, locality)
    dom = alt_impl["dominant"]
    findings = []
    if ratios["alt_impl_backward_forward"] >= MEANINGFUL_RATIO:
        findings.append(
            "backward fp16_alt_impl flag (0x4) slows the dominant projection "
            f"{ratios['alt_impl_backward_forward']:.2f}x vs forward (flags=0); "
            "D71 swept only flags=0 and never exercised this path"
        )
    if dom["solution_sets_differ"]:
        findings.append("fp16_alt_impl admits a different Tensile solution set")
    if ratios["sustained_degradation"] >= MEANINGFUL_RATIO:
        findings.append(
            f"sustained-load throughput degrades {ratios['sustained_degradation']:.2f}x "
            "(thermal/clock throttling)"
        )
    if ratios["locality_cold_hot"] >= MEANINGFUL_RATIO:
        findings.append(
            f"cold pooled inputs are {ratios['locality_cold_hot']:.2f}x slower than hot "
            "(memory locality)"
        )
    if ratios["locality_fresh_hot"] >= MEANINGFUL_RATIO:
        findings.append(
            f"freshly written inputs are {ratios['locality_fresh_hot']:.2f}x slower than hot "
            "(dirty-cache / write-then-read locality)"
        )
    candidates = {
        "alt_impl_flag": ratios["alt_impl_backward_forward"],
        "thermal_throttling": ratios["sustained_degradation"],
        "memory_locality": max(
            ratios["locality_cold_hot"], ratios["locality_fresh_hot"]
        ),
    }
    primary = max(candidates, key=candidates.get)
    if candidates[primary] < MEANINGFUL_RATIO:
        attribution = "not_reproduced_by_isolated_fixtures"
        primary = None
    elif candidates[primary] >= DOMINANT_RATIO:
        attribution = f"{primary}_dominant"
    else:
        attribution = f"{primary}_contributor"
    return dict(
        status="passed",
        dominant_layout=dom["rocblas_layout"],
        dominant_args=dom["args"],
        ratios=ratios,
        candidates=candidates,
        primary_cause=primary,
        attribution=attribution,
        findings=findings,
        sustained_clock_visible=sustained["clock_visible"],
        sustained_temp_c_max=sustained["temp_c_max"],
        thresholds={
            "meaningful": MEANINGFUL_RATIO,
            "dominant": DOMINANT_RATIO,
        },
    )
