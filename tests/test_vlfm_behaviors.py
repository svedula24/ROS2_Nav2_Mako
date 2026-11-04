"""The shared move executor's pure parts (m20_vlmnav.behaviors): the time allowance and the clamped duration."""
import math

from m20_vlmnav import moves
from m20_vlmnav.behaviors import MAX_ALLOWANCE_S, MOVE_MARGIN_S, SPIN_RATE, allowance_s, duration_fields


def test_allowance_is_twice_the_ideal_plus_the_margin():
    assert allowance_s(moves.Move(moves.FORWARD, 3.0), 0.3) == 3.0 / 0.3 * 2 + MOVE_MARGIN_S          # 25 s
    assert abs(allowance_s(moves.Move(moves.TURN_LEFT, math.pi / 2), 0.3) - (math.pi / 2 / SPIN_RATE * 2 + MOVE_MARGIN_S)) < 1e-9
    assert allowance_s(moves.Move(moves.TURN_RIGHT, math.pi), 0.3) == allowance_s(moves.Move(moves.TURN_LEFT, math.pi), 0.3)


def test_duration_fields_are_clamped_and_split():
    assert duration_fields(7.25) == (7, 250000000)
    assert duration_fields(-3.0) == (0, 0)
    assert duration_fields(1e6) == (int(MAX_ALLOWANCE_S), 0)
