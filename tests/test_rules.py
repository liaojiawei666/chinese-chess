import unittest

from engine import (
    ACTION_SPACE_SIZE,
    Color,
    GameStatusReason,
    GameState,
    Move,
    PieceKind,
    Position,
    action_id_to_move,
    move_to_action_id,
)


class XiangqiRulesTest(unittest.TestCase):
    def test_starting_position_round_trips_and_has_expected_moves(self) -> None:
        position = Position.starting()

        self.assertEqual(position.to_fen(), "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r")
        self.assertEqual(len(position.legal_moves()), 44)
        self.assertEqual(sum(position.legal_move_mask()), 44)
        self.assertEqual(len(position.legal_move_mask()), ACTION_SPACE_SIZE)

    def test_action_id_round_trip(self) -> None:
        move = Move(1, 9, 2, 7)

        self.assertEqual(action_id_to_move(move_to_action_id(move)), move)

    def test_make_move_returns_new_position(self) -> None:
        position = Position.starting()
        move = Move(1, 9, 2, 7)

        next_position = position.make_move(move)

        self.assertIsNone(position.piece_at(2, 7))
        moved_piece = next_position.piece_at(2, 7)
        self.assertIsNotNone(moved_piece)
        self.assertEqual(moved_piece.color, Color.RED)
        self.assertEqual(moved_piece.kind, PieceKind.HORSE)
        self.assertIsNone(next_position.piece_at(1, 9))
        self.assertEqual(next_position.side_to_move, Color.BLACK)

    def test_horse_leg_blocks_move(self) -> None:
        position = Position.from_fen("4k4/9/9/9/9/9/4P4/4H4/9/4K4 r")

        self.assertNotIn(Move(4, 7, 5, 5), position.legal_moves())
        self.assertIn(Move(4, 7, 6, 8), position.legal_moves())

    def test_elephant_cannot_cross_or_jump_eye(self) -> None:
        blocked_eye = Position.from_fen("4k4/9/9/9/9/9/9/9/3P5/2E1K4 r")
        river_edge = Position.from_fen("4k4/9/9/9/9/2E6/9/9/9/4K4 r")

        self.assertNotIn(Move(2, 9, 4, 7), blocked_eye.legal_moves())
        self.assertNotIn(Move(2, 5, 4, 3), river_edge.legal_moves())

    def test_advisor_stays_inside_palace(self) -> None:
        position = Position.from_fen("4k4/9/9/9/9/9/9/9/3A5/4K4 r")

        self.assertIn(Move(3, 8, 4, 7), position.legal_moves())
        self.assertNotIn(Move(3, 8, 2, 7), position.legal_moves())

    def test_pawn_moves_sideways_only_after_crossing_river(self) -> None:
        before_river = Position.from_fen("k8/9/9/9/9/9/4P4/9/9/4K4 r")
        after_river = Position.from_fen("k8/9/9/9/4P4/9/9/9/9/4K4 r")

        self.assertEqual(
            {move.target for move in before_river.legal_moves() if move.source == (4, 6)},
            {(4, 5)},
        )
        self.assertEqual(
            {move.target for move in after_river.legal_moves() if move.source == (4, 4)},
            {(4, 3), (3, 4), (5, 4)},
        )

    def test_cannon_needs_exactly_one_screen_to_capture(self) -> None:
        position = Position.from_fen("4k4/9/9/9/4p4/9/4P4/9/4C4/4K4 r")

        self.assertIn(Move(4, 8, 4, 4), position.legal_moves())
        self.assertNotIn(Move(4, 8, 4, 0), position.legal_moves())

    def test_flying_general_is_check_and_blocks_exposing_own_king(self) -> None:
        position = Position.from_fen("4k4/9/4R4/9/9/9/9/9/9/4K4 r")

        self.assertFalse(position.is_in_check(Color.RED))
        self.assertNotIn(Move(4, 2, 3, 2), position.legal_moves())
        self.assertNotIn(Move(4, 2, 4, 0), position.legal_moves())
        self.assertIn(Move(4, 2, 4, 1), position.legal_moves())

    def test_checkmate_status(self) -> None:
        position = Position.from_fen("3k5/4R4/3R5/9/9/9/9/9/9/4K4 b")

        status = position.status()

        self.assertTrue(status.is_terminal)
        self.assertEqual(status.reason, GameStatusReason.CHECKMATE)
        self.assertEqual(status.winner, Color.RED)

    def test_repetition_key_ignores_counters(self) -> None:
        first = Position.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r - - 0 1")
        later = Position.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r - - 8 20")

        self.assertEqual(first.repetition_key(), later.repetition_key())

    def test_piece_indexes_follow_moves(self) -> None:
        position = Position.starting()
        moving_piece = position.piece_at(1, 9)
        self.assertIsNotNone(moving_piece)
        moving_piece_id = moving_piece.piece_id

        next_position = position.make_move(Move(1, 9, 2, 7))

        moved_piece = next_position.piece_at(2, 7)
        self.assertIsNotNone(moved_piece)
        self.assertIsNone(next_position.piece_at(1, 9))
        self.assertEqual(moved_piece.piece_id, moving_piece_id)
        self.assertEqual(next_position.piece_positions[moving_piece_id], (2, 7))
        self.assertEqual(next_position.king_square_by_color[Color.RED], (4, 9))

    def test_game_state_detects_perpetual_check(self) -> None:
        game = GameState.from_position(Position.from_fen("4k4/9/3R5/9/9/9/9/9/4A4/4K4 r"))

        cycle = (
            Move(3, 2, 4, 2),
            Move(4, 0, 3, 0),
            Move(4, 2, 3, 2),
            Move(3, 0, 4, 0),
        )
        for move in cycle * 2:
            game = game.make_move(move)

        status = game.status()
        self.assertTrue(status.is_terminal)
        self.assertEqual(status.reason, GameStatusReason.PERPETUAL_CHECK)
        self.assertEqual(status.winner, Color.BLACK)

    def test_game_state_detects_perpetual_chase(self) -> None:
        game = GameState.from_position(Position.from_fen("k3h3h/9/9/9/4R4/9/9/9/9/1K7 r"))

        cycle = (
            Move(4, 4, 4, 5),
            Move(8, 0, 7, 2),
            Move(4, 5, 4, 4),
            Move(7, 2, 8, 0),
        )
        for move in cycle * 2:
            game = game.make_move(move)

        status = game.status()
        self.assertTrue(status.is_terminal)
        self.assertEqual(status.reason, GameStatusReason.PERPETUAL_CHASE)
        self.assertEqual(status.winner, Color.BLACK)

    def test_game_state_detects_mutual_perpetual(self) -> None:
        game = GameState.from_position(Position.from_fen("k3h4/8r/9/9/4R4/9/9/9/9/1K6H r"))

        cycle = (
            Move(4, 4, 4, 5),
            Move(8, 1, 8, 2),
            Move(4, 5, 4, 4),
            Move(8, 2, 8, 1),
        )
        for move in cycle * 2:
            game = game.make_move(move)

        status = game.status()
        self.assertTrue(status.is_terminal)
        self.assertEqual(status.reason, GameStatusReason.MUTUAL_PERPETUAL)
        self.assertIsNone(status.winner)


if __name__ == "__main__":
    unittest.main()
