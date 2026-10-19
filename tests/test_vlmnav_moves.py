"""Real-time B move vocabulary: VLM text -> move, and which frames the VLM sees."""
import math

from m20_vlmnav import history, moves
from m20_vlmnav.moves import FORWARD, STOP, TURN_LEFT, TURN_RIGHT, Move


def test_parse_the_four_moves_and_their_amounts():
    assert moves.parse("move forward 75 cm") == Move(FORWARD, 0.75)
    assert moves.parse("turn left 30 degrees") == Move(TURN_LEFT, math.radians(30))
    assert moves.parse("turn right 15 degrees") == Move(TURN_RIGHT, math.radians(15))
    assert moves.parse("stop") == Move(STOP)


def test_parse_tolerates_phrasing_but_not_missing_meaning():
    assert moves.parse("The next action is to move forward 0.75 m.") == Move(FORWARD, 0.75)
    assert moves.parse("Turn right by 45°") == Move(TURN_RIGHT, math.radians(45))
    assert moves.parse("Stop.") == Move(STOP)
    for bad in ("", None, "move forward", "turn left", "move forward 0 cm", "I see a ladder", "turn around"):
        assert moves.parse(bad) is None, bad


def test_backward_is_not_a_move():
    """No backward move (the camera can't see behind the robot): any reverse request matches nothing -> None, so the
    episode ends instead of the robot reversing blind (or a "back" being misread as forward)."""
    for reply in ("move backward 50 cm", "move backwards 0.5 m", "go back 25 centimeters", "back up 30 cm",
                  "reverse 40 cm"):
        assert moves.parse(reply) is None, reply
    assert not hasattr(moves, "BACKWARD")


def test_two_moves_in_one_reply_is_ambiguous_not_guessed():
    """A reply naming more than one move parses as None (the episode ends rather than picking one)."""
    assert moves.parse("move forward 50 cm and then stop") is None
    assert moves.parse("stop, do not turn left 30 degrees") is None


def test_reasoning_is_ignored_and_an_answer_block_wins():
    reasoned = "<think>I should not stop yet; turn left 30 degrees looks right</think>turn left 30 degrees"
    assert moves.parse(reasoned) == Move(TURN_LEFT, math.radians(30))
    assert moves.parse("<think>maybe stop? no, the ladder is ahead so move") is None    # truncated reasoning only
    assert moves.parse("<think>stop or go</think><answer>move forward 50 cm</answer> stop") == Move(FORWARD, 0.5)
    assert moves.parse("turn right 45º") == Move(TURN_RIGHT, math.radians(45))        # ordinal sign typed for degrees


def test_sanity_clamp_is_applied_and_flagged():
    big = moves.parse("move forward 50 m")
    assert big == Move(FORWARD, moves.MAX_DIST_M, True) and big.clamped
    spin = moves.parse("turn left 3600 degrees")
    assert spin.amount == moves.MAX_TURN_RAD and spin.clamped
    assert not moves.parse("move forward 3 m").clamped


def test_move_text_round_trips_and_spin_sign():
    for m in (Move(FORWARD, 0.75), Move(TURN_LEFT, math.radians(30)), Move(TURN_RIGHT, math.radians(15)), Move(STOP)):
        assert moves.parse(m.text()).text() == m.text()
    assert Move(TURN_LEFT, 0.5).yaw() == 0.5 and Move(TURN_RIGHT, 0.5).yaw() == -0.5 and Move(FORWARD, 1).yaw() == 0


def test_cap_flag_is_recorded_not_applied():
    assert moves.cap_would_trigger(Move(FORWARD, 1.5)) and not moves.cap_would_trigger(Move(FORWARD, 1.0))
    assert moves.cap_would_trigger(Move(TURN_LEFT, math.radians(90)))
    assert not moves.cap_would_trigger(Move(TURN_RIGHT, math.radians(45)))
    assert not moves.cap_would_trigger(Move(STOP))
    assert moves.parse("move forward 2.5 m").amount == 2.5        # parsed in full: the 1 m cap is not applied


def test_history_picks_current_plus_seven_evenly_spaced():
    assert history.pick(0) == []
    assert history.pick(1) == [0] * 8
    assert history.pick(3) == [0, 0, 0, 0, 0, 0, 1, 2]
    assert history.pick(8) == list(range(8))
    p = history.pick(64)
    assert len(p) == 8 and p[0] == 0 and p[-1] == 63 and p == sorted(p)
    gaps = [b - a for a, b in zip(p, p[1:])]
    assert max(gaps) - min(gaps) <= 1                              # evenly spaced
