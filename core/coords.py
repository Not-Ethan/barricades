N = 9  # board is N x N cells

_FILES = "abcdefghi"


def on_board(cell):
    x, y = cell
    return 0 <= x < N and 0 <= y < N


def cell_to_str(cell):
    x, y = cell
    return f"{_FILES[x]}{y + 1}"


def str_to_cell(s):
    return (_FILES.index(s[0]), int(s[1:]) - 1)
