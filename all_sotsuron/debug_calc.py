
import math

dist = 0.05
velo = 10
num_points_large = 100000
freq_large = velo / (num_points_large * dist)
print(f"Freq (100k): {freq_large}")
print(f"Sampling (100k, raw): {freq_large * num_points_large}")
print(f"Sampling (100k, wrapped 34464): {freq_large * 34464}")

num_points_50k = 50000
freq_50k = velo / (num_points_50k * dist)
print(f"Freq (50k): {freq_50k}")
print(f"Sampling (50k, raw): {freq_50k * num_points_50k}")

rounded_freq = round(freq_50k, 3)
print(f"Rounded Freq (3 decimal): {rounded_freq}")
print(f"Sampling (50k, rounded): {rounded_freq * num_points_50k}")

exact_sampling = 200.00002
print(f"Reverse engineering freq from 200.00002/50000: {exact_sampling/num_points_50k}")
