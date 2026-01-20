import pickle
import numpy as np

# Load the list from the pickle file
with open('abcrown_vnnlib.pkl', 'rb') as f:
    abcrown_vnnlib = pickle.load(f)

with open('robxpl_vnnlib.pkl', 'rb') as f:
    robxpl_vnnlib = pickle.load(f)

print(abcrown_vnnlib)
print(robxpl_vnnlib)

