import sys

filename = sys.argv[1]
target = "Average_F"

with open(filename,'r') as f:
    scores = [float(line[22:28]) for line in f if target in line]

names = ["R1", "R2", "RL"]


for name, score in zip(names, scores):
    print(name + ": %.4g" % (score * 100))
