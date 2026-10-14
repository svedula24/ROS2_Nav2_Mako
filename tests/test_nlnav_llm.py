"""llm.py: schema validation and backend plumbing (no network, no model)."""
import json

import pytest

from m20_nlnav import llm


def test_validate_accepts_the_artifact_examples():
    q = llm.validate({"steps": [{"target": "the valve", "anchor": "Big Yellow Tank", "relation": "near",
                                 "pick": "best"}, {"target": "dock", "pick": "best"}], "stop": False})
    assert [s.target for s in q.steps] == ["valve", "dock"]
    assert q.steps[0].anchor == "big yellow tank" and q.steps[0].relation == "near"
    assert llm.validate({"steps": [], "stop": True}).stop


@pytest.mark.parametrize("bad", [
    None, {}, {"steps": "ladder", "stop": False}, {"steps": [], "stop": False},
    {"steps": [{"target": "", "pick": "best"}], "stop": False},
    {"steps": [{"target": "valve", "relation": "above", "anchor": "pump", "pick": "best"}], "stop": False},
    {"steps": [{"target": "valve", "pick": "random"}], "stop": False},
    {"understood": False, "steps": [{"target": "back", "pick": "best"}], "stop": False},   # gibberish, model guessed
])
def test_validate_rejects(bad):
    with pytest.raises(llm.NotUnderstood):
        llm.validate(bad)


def test_json_backend_passthrough_and_parse():
    q = llm.parse(json.dumps({"steps": [{"target": "ladder", "pick": "nearest"}], "stop": False}),
                  llm.JsonBackend())
    assert q.steps[0].target == "ladder" and q.steps[0].pick == "nearest"
    with pytest.raises(llm.NotUnderstood):
        llm.parse("go to the ladder", llm.JsonBackend())


class Boom:
    name = "boom"

    def complete_json(self, *a, **k):
        raise TimeoutError("model not loaded")


def test_backend_failure_is_not_understood_not_a_crash():
    with pytest.raises(llm.ParserUnavailable) as e:          # a NotUnderstood, but reported as our error
        llm.parse("go to the ladder", Boom())
    assert "TimeoutError" in str(e.value) and isinstance(e.value, llm.NotUnderstood)


@pytest.mark.parametrize("text", ["stop", "STOP!", " halt ", "cancel", "please stop", "stop the robot", "abort."])
def test_stop_is_recognised_without_the_llm(text):
    assert llm.is_stop(text)


@pytest.mark.parametrize("text", ["", "go to the stop sign", "stop by the printer", "don't stop", "stopwatch"])
def test_stop_fast_path_ignores_navigation(text):
    assert not llm.is_stop(text)


def test_system_prompt_lists_named_places():
    seen = {}

    class Spy:
        name = "spy"

        def complete_json(self, system, user, schema, timeout_s=0):
            seen["system"], seen["schema"] = system, schema
            return {"steps": [{"target": "dock", "pick": "best"}], "stop": False}
    llm.parse("go to the dock", Spy(), places=["start", "dock"])
    assert "start, dock" in seen["system"] and seen["schema"] is llm.SCHEMA


def test_understood_flag_is_optional_and_true_passes():
    assert llm.validate({"understood": True, "steps": [{"target": "ladder", "pick": "best"}], "stop": False})
    assert llm.validate({"understood": False, "steps": [], "stop": True}).stop     # "stop" always wins
    assert list(llm.SCHEMA["properties"])[0] == "understood"                        # decided before the steps


def test_relation_without_anchor_is_dropped_not_rejected():
    """qwen2.5:3b answers "the nearest trash can" with relation "near" and no anchor; the step is still a valid
    target, so the meaningless relation goes and the query stands (found by tools/nlnav_eval, 2026-10-14)."""
    q = llm.validate({"understood": True, "steps": [{"target": "trash can", "anchor": None, "relation": "near",
                                                     "pick": "nearest"}], "stop": False})
    assert q.steps[0].target == "trash can" and q.steps[0].relation is None and q.steps[0].pick == "nearest"
