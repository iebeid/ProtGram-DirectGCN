def add_matrices(matrix_a, matrix_b):
    """
    Adds two matrices together.

    Args:
        matrix_a: The first matrix (list of lists).
        matrix_b: The second matrix (list of lists).

    Returns:
        A new matrix that is the sum of the two input matrices,
        or None if they cannot be added.
    """
    # --- Verify dimensions for addition ---
    rows_a = len(matrix_a)
    if rows_a == 0:
        return [] if len(matrix_b) == 0 else None
    cols_a = len(matrix_a[0])

    rows_b = len(matrix_b)
    if rows_b == 0:
        return None
    cols_b = len(matrix_b[0])

    if rows_a != rows_b or cols_a != cols_b:
        print("Error: Matrices must have the same dimensions to be added.")
        return None

    # --- Calculate the sum matrix ---
    sum_matrix = [[0 for _ in range(cols_a)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_a):
            sum_matrix[i][j] = matrix_a[i][j] + matrix_b[i][j]

    return sum_matrix


def is_symmetric(matrix):
    """
    Checks if a single matrix is symmetric.

    Args:
        matrix: The matrix to check (list of lists).

    Returns:
        True if the matrix is symmetric, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix is considered symmetric

    cols = len(matrix[0])

    # --- A matrix must be square to be symmetric ---
    if rows != cols:
        return False

    # --- Check if matrix[i][j] == matrix[j][i] ---
    for i in range(rows):
        for j in range(i + 1, cols):  # Only need to check the upper triangle
            if matrix[i][j] != matrix[j][i]:
                return False

    return True


def is_addition_symmetric(matrix_a, matrix_b):
    """
    Checks if the sum of two matrices results in a symmetric matrix by
    combining add_matrices and is_symmetric.

    Args:
        matrix_a: The first matrix (list of lists).
        matrix_b: The second matrix (list of lists).

    Returns:
        True if the matrices can be added and their sum is symmetric, False otherwise.
    """
    sum_result = add_matrices(matrix_a, matrix_b)

    # If addition failed (e.g., different dimensions), sum_result will be None
    if sum_result is None:
        return False

    return is_symmetric(sum_result)


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
out_edges = [[0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0, 0, 0]]

in_edges = [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]

print(f"Is out_edges symmetric? {is_symmetric(out_edges)}")
print(f"Is in_edges symmetric? {is_symmetric(in_edges)}")

print(f"\nIs the sum of out_edges and in_edges symmetric? {is_addition_symmetric(out_edges, in_edges)}")

if are_transposes(out_edges, in_edges):
    print("in_edges is the transpose of out_edges.")
else:
    print("in_edges is NOT the transpose of out_edges.")
