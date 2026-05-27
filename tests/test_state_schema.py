from amr_swarm.state_schema import CriticDecision, ResearcherOutput


def test_researcher_output_defaults():
    out = ResearcherOutput(summary="Tóm tắt thị trường.")
    assert out.sources == []


def test_critic_decision_literal():
    d = CriticDecision(decision="APPROVE", feedback="OK")
    assert d.decision == "APPROVE"
