from scripts.register_model import registration_decision


def _passing_report() -> dict:
    row = {"policy": "thompson_sampling", "seeds": 30, "clip_rate": 0.01}
    return {
        "claim": {
            "simulation_uplift_ci_above_zero": True,
            "ope_clip_rate_below_10_percent": True,
            "model_ranking_matches_observed_test": True,
            "deployment_claim": True,
        },
        "test_observed_roc_auc": 0.7,
        "simulation": [row],
        "ope": [row],
    }


def test_registration_blocks_the_current_candidate_shape() -> None:
    decision = registration_decision(
        {
            "claim": {
                "simulation_uplift_ci_above_zero": True,
                "ope_clip_rate_below_10_percent": False,
                "model_ranking_matches_observed_test": False,
                "deployment_claim": False,
            },
            "test_observed_roc_auc": 0.41,
            "simulation": [{"policy": "thompson_sampling", "seeds": 30}],
            "ope": [{"policy": "thompson_sampling", "clip_rate": 0.31}],
        }
    )
    assert decision["registered"] is False
    assert decision["promotion_status"] == "blocked"
    assert decision["failures"]


def test_registration_stays_unregistered_when_the_gate_passes() -> None:
    decision = registration_decision(_passing_report())
    assert decision["failures"] == []
    assert decision["promotion_status"] == "pending_review"
    assert decision["registered"] is False
