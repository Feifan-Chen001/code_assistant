
import pickle
import os

# Security issue: pickle untrusted data
data = pickle.loads(user_input)

# Security issue: hardcoded password
password = "admin123"
