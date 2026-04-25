import time
from functools import wraps

def with_spinner(text="Ładowanie..."):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import streamlit as st
                with st.spinner(text):
                    return func(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} wykonało się w {end-start:.2f}s")
        return result
    return wrapper
