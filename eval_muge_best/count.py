import sys

pth = sys.argv[1]
# pth = "/workspace/EDMamba/output-VM/0806-BSDS-stageII-rand-test/epoch-11-checkpoint-ss/record.txt"

with open(pth) as f:
    files = f.readlines()

index = []
for i in files[:-1]:
     index.append(float(i.strip().split("\t")[-1]))

from collections import Counter

count = Counter(index)
print(count)

