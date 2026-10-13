import ast
import os

from m20_nlnav import tier0
from m20_nlnav.world import DETECTOR_CLASSES
from nlnav_fixtures import office

W = office()
K = DETECTOR_CLASSES


def p(text):
    return tier0.parse(text, K, W.waypoints)


def test_detector_classes_in_sync_with_detector_node():
    src = os.path.join(os.path.dirname(__file__), "..", "tools", "splat", "detector_node.py")
    tree = ast.parse(open(src, encoding="utf-8").read())
    office_list = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "OFFICE")
    assert office_list == DETECTOR_CLASSES


def test_object_phrasings():
    for text in ["go to the printer", "Go to the printer.", "please take me to the printer",
                 "drive to a printer", "navigate to printer", "head to the printer now"]:
        c = p(text)
        assert c and c.action == "goto" and len(c.steps) == 1, text
        assert c.steps[0].kind == "object" and c.steps[0].classes == ["printer"], text


def test_synonyms_and_plurals():
    assert p("go to the bin").steps[0].classes == ["trash can"]
    assert p("go to the chair").steps[0].classes == ["office chair"]
    assert p("go to the chairs").steps[0].classes == ["office chair"]
    assert set(p("go to the cabinet").steps[0].classes) == {"filing cabinet", "storage cabinet"}


def test_nearest():
    t = p("take me to the nearest trash can").steps[0]
    assert t.nearest and t.classes == ["trash can"]
    assert p("go to the closest chair").steps[0].nearest


def test_waypoints_and_number_words():
    for text in ["go to stop 3", "go to stop three", "go to Stop3", "go to WP12", "go to wp 12"]:
        t = p(text).steps[0]
        assert t.kind == "waypoint", text
    assert p("go to stop three").steps[0].name == "stop 3"


def test_start_and_back():
    for text in ["go back", "come back", "go home", "return", "go back to the start",
                 "go to the start", "return home"]:
        c = p(text)
        assert c and [s.kind for s in c.steps] == ["start"], text


def test_multi_step():
    c = p("go to the printer, then the nearest chair and then come back")
    assert [s.kind for s in c.steps] == ["object", "object", "back"]
    assert c.steps[1].nearest
    c = p("go to stop 3 then go to the printer then go home")
    assert [s.kind for s in c.steps] == ["waypoint", "object", "start"]


def test_stop():
    for text in ["stop", "Stop!", "cancel", "halt"]:
        assert p(text).action == "stop", text


def test_rejects_what_it_does_not_understand():
    # never trims words it doesn't know: these are for a later (LLM/VLM) tier
    for text in ["go to the red chair", "go to the chair next to the whiteboard",
                 "go to the coffee machine", "dance", "go to the second desk on the left",
                 "go to the printer then dance", ""]:
        assert p(text) is None, text
