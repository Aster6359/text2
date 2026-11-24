from collections import Counter
from math import log2


def entropy(pixels):
    counts = Counter(pixels)
    probabilities = [count / len(pixels) for count in counts.values()]
    return -sum(probability * log2(probability) for probability in probabilities)


datasets = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 2, 2, 1, 3, 3],
]

for index, pixels in enumerate(datasets, start=1):
    print(f"第 {index} 组：{entropy(pixels):.4f} bits")

reordered = [0, 3, 1, 2, 3, 0, 2, 1]
print(f"第三组重新排列后：{entropy(reordered):.4f} bits")