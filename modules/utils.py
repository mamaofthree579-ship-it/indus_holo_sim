import streamlit as st

# ------------------------------------------------------------------------------
# Safe Execution Wrapper
# ------------------------------------------------------------------------------
def safe_run(label, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{label} failed: {e}")
        return None


# ------------------------------------------------------------------------------
# Validate Matrix
# ------------------------------------------------------------------------------
def validate_matrix(matrix):
    if matrix is None:
        raise ValueError("Matrix is None")
    if hasattr(matrix, "empty") and matrix.empty:
        raise ValueError("Matrix is empty")
