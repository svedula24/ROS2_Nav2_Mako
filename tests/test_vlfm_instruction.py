"""The search's front door (m20_vlmnav.instruction): moves, object, or neither. No backward move exists."""
import math

from m20_vlmnav import moves
from m20_vlmnav.instruction import MOVES, OBJECT, UNKNOWN, classify


def kinds(text):
    return [(m.kind, round(m.amount, 3)) for m in classify(text).seq]


def test_single_moves_and_sequences_are_moves():
    assert classify("turn left 90 degrees").kind == MOVES
    assert kinds("turn left 90 degrees") == [(moves.TURN_LEFT, round(math.pi / 2, 3))]
    assert kinds("move forward 2 meters, then stop") == [(moves.FORWARD, 2.0)]
    assert kinds("turn around and walk 2 meters, then stop") == [(moves.TURN_LEFT, round(math.pi, 3)), (moves.FORWARD, 2.0)]
    assert kinds("go 50 cm then turn right 45 degrees") == [(moves.FORWARD, 0.5), (moves.TURN_RIGHT, round(math.radians(45), 3))]


def test_object_instructions_become_a_phrase():
    assert classify("go to the ladder") == classify("ladder") or classify("go to the ladder").phrase == "ladder"
    assert classify("find a trash can and stop in front of it").phrase == "trash can"
    assert classify("refrigerator").kind == OBJECT


def test_a_long_typed_distance_is_split_into_legs_not_shortened():
    assert kinds("walk 7 meters") == [(moves.FORWARD, 3.0), (moves.FORWARD, 3.0), (moves.FORWARD, 1.0)]
    assert kinds("move forward 5 m then stop") == [(moves.FORWARD, 3.0), (moves.FORWARD, 2.0)]
    assert classify("walk 500 meters").kind == UNKNOWN          # above MAX_TYPED_DIST_M: refused, not shortened


def test_a_conditional_stop_refuses_the_whole_instruction():
    # "stop at the door" needs a camera; the moves before it must not run blind
    assert classify("move forward 2 meters, then stop at the door").kind == UNKNOWN
    assert classify("turn left 90 degrees and stop next to the chair").kind == UNKNOWN
    assert kinds("move forward 2 meters, then stop here") == [(moves.FORWARD, 2.0)]


def test_article_less_and_can_of_phrases_are_objects():
    assert classify("find ladder").phrase == "ladder"
    assert classify("go to refrigerator").phrase == "refrigerator"
    assert classify("can of soda").phrase == "can of soda"
    assert classify("is there a ladder").kind == UNKNOWN
    assert classify("can you find the ladder").kind == UNKNOWN


def test_routes_questions_and_backward_moves_are_unknown():
    assert classify("go straight down the corridor and stop next to the first desk on your left").kind == UNKNOWN
    assert classify("stop").kind == UNKNOWN                   # a stop alone is not a journey (the cancel path is)
    assert classify("what time is it").kind == UNKNOWN
    # no backward move exists in Real-time B (removed 2026-10-15: the camera can't see behind the robot)
    assert classify("move backward 50 cm").kind == UNKNOWN
    assert classify("back up 1 meter").kind == UNKNOWN
    assert classify("turn around and move backward 50 cm").kind == UNKNOWN
