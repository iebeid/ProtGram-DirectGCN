def are_transposes(matrix_a, matrix_b):
    """
    Checks if two matrices are the transpose of each other.

    Args:
        matrix_a: The first matrix (list of lists).
        matrix_b: The second matrix (list of lists).

    Returns:
        True if matrix_b is the transpose of matrix_a, False otherwise.
    """
    # Get dimensions of matrix_a
    rows_a = len(matrix_a)
    if rows_a == 0:
        return len(matrix_b) == 0  # Both are empty matrices
    cols_a = len(matrix_a[0])

    # Get dimensions of matrix_b
    rows_b = len(matrix_b)
    if rows_b == 0:
        return False  # A is not empty, B is
    cols_b = len(matrix_b[0])

    # Check if dimensions are reversed
    if rows_a != cols_b or cols_a != rows_b:
        return False

    # Check if elements are transposed
    for i in range(rows_a):
        for j in range(cols_a):
            if matrix_a[i][j] != matrix_b[j][i]:
                return False

    return True

# --- Example Usage ---

# Example 1: Matrices that are transposes
out_edges = [
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0]
]

in_edges = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]
]

if are_transposes(out_edges, in_edges):
    print("in_edges is the transpose of out_edges.")
else:
    print("in_edges is NOT the transpose of out_edges.")