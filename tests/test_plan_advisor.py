from app.config import SOURCE_DIR
from app.ingest import ingest
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, suggest_exploration_plan


def test_exploration_draft_is_differentiated_and_requires_confirmation(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    draft = suggest_exploration_plan(registry, n_cases=3)
    assert draft.plan_id.startswith("draft-")
    assert len({case.label for case in draft.cases}) == 3
    assert all(case.label.startswith("explore-") for case in draft.cases)
    stored = PlanStore(tmp_path / "plans.sqlite3").create(draft)
    assert stored["status"] == "pending"
