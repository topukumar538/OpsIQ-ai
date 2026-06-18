# Location: backend/tests/conftest.py
import sys
import os

# Add backend/ to the Python path so tests can import
# auth, router, postmortem etc. directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))