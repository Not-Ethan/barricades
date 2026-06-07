from core.coords import N, cell_to_str, str_to_cell, on_board


def test_cell_to_str_corners():
    assert cell_to_str((0, 0)) == "a1"
    assert cell_to_str((4, 0)) == "e1"
    assert cell_to_str((4, 8)) == "e9"
    assert cell_to_str((8, 8)) == "i9"


def test_str_to_cell_roundtrip():
    for cell in [(0, 0), (4, 0), (4, 8), (8, 8), (3, 5)]:
        assert str_to_cell(cell_to_str(cell)) == cell


def test_on_board():
    assert on_board((0, 0))
    assert on_board((8, 8))
    assert not on_board((-1, 0))
    assert not on_board((9, 0))
    assert not on_board((0, 9))


def test_board_size():
    assert N == 9
