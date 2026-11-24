import csv
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).eval()
categories = weights.meta["categories"]
category_index = {name: index for index, name in enumerate(categories)}

data = Imagenette(
    root="data",
    split="val",
    size="160px",
    download=True,
    transform=weights.transforms(),
)
label_map = torch.tensor([category_index[names[0]] for names in data.classes])

indices = random.Random(42).sample(range(len(data)), 300)
loader = DataLoader(Subset(data, indices), batch_size=16, num_workers=0)

rows = []
all_probabilities = []
with torch.inference_mode():
    for images, labels in loader:
        true_ids = label_map[labels]
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        confidence, predicted_ids = probabilities.max(dim=1)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=1)
        entropy = entropy / torch.log(torch.tensor(2.0))
        all_probabilities.extend(probabilities.unbind())

        for true_id, predicted_id, score, bits in zip(
            true_ids.tolist(), predicted_ids.tolist(),
            confidence.tolist(), entropy.tolist(),
        ):
            rows.append({
                "true_label": categories[true_id],
                "prediction": categories[predicted_id],
                "confidence": score,
                "entropy_bits": bits,
                "correct": int(true_id == predicted_id),
            })

for position, row in enumerate(rows):
    row["dataset_index"] = indices[position]
    row["position"] = position

rows.sort(key=lambda row: row["entropy_bits"], reverse=True)
with Path("predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

group_size = len(rows) // 5
for name, group in [
    ("熵最高的 20%", rows[:group_size]),
    ("熵最低的 20%", rows[-group_size:]),
]:
    errors = sum(1 - row["correct"] for row in group)
    print(f"{name}：错 {errors}/{len(group)} 张，错误率 {errors / len(group):.1%}")

examples = [("Highest entropy", rows[0]), ("Lowest entropy", rows[-1])]
with Path("mobilenet_distributions.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["dataset_index", "true_label", "category", "probability"])
    for row in rows:
        probabilities = all_probabilities[row["position"]]
        for category, probability in zip(categories, probabilities.tolist()):
            writer.writerow([row["dataset_index"], row["true_label"], category, probability])

for name, row in examples:
    values, category_ids = all_probabilities[row["position"]].topk(5)
    print(f"\n{name}：图片编号 {row['dataset_index']}，真实类别 {row['true_label']}，熵 {row['entropy_bits']:.4f} bits")
    for category_id, value in zip(category_ids.tolist(), values.tolist()):
        print(f"  {categories[category_id]}: {value:.2%}")
    print(f"  其余 995 类合计：{1 - values.sum().item():.2%}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("安装 matplotlib 后可生成图表：python -m pip install matplotlib")
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    representatives = [rows[-1], rows[len(rows) // 2], rows[0]]
    for name, row in zip(["最低预测熵", "中间预测熵", "最高预测熵"], representatives):
        probabilities = all_probabilities[row["position"]]
        ordered = probabilities.sort(descending=True).values
        axes[0].plot(range(1, len(ordered) + 1), ordered.cumsum(0).tolist(),
                     label=f"{name}: H={row['entropy_bits']:.2f} bits")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("按概率从高到低累计的类别数（对数刻度）")
    axes[0].set_ylabel("累计概率")
    axes[0].set_title("三张图片的预测概率分布")
    axes[0].legend()

    sorted_rows = sorted(rows, key=lambda row: row["entropy_bits"])
    groups = [sorted_rows[start:start + group_size] for start in range(0, len(rows), group_size)]
    group_labels = [f"{group[0]['entropy_bits']:.2f}–{group[-1]['entropy_bits']:.2f}" for group in groups]
    error_rates = [sum(1 - row["correct"] for row in group) / len(group) for group in groups]
    axes[1].bar(group_labels, error_rates)
    axes[1].set_xlabel("预测熵范围（比特）")
    axes[1].set_ylabel("错误率")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("不同预测熵分组的错误率（每组 60 张）")
    axes[1].tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig("mobilenet_distributions.png", dpi=150)
    plt.close(figure)
    print("已保存 mobilenet_distributions.png 和 mobilenet_distributions.csv")

    raw_data = Imagenette(root="data", split="val", size="160px", download=False)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, (name, row) in enumerate(zip(["最低", "中间", "最高"], representatives)):
        image, _ = raw_data[row["dataset_index"]]
        histogram = image.convert("L").histogram()
        counts = [sum(histogram[start:start + 16]) for start in range(0, 256, 16)]
        total = sum(counts)
        probabilities = [count / total for count in counts]
        pixel_entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability > 0)
        axes[0, column].imshow(image)
        axes[0, column].set_title(f"{name}预测熵\nH={row['entropy_bits']:.2f} 比特")
        axes[0, column].axis("off")
        axes[1, column].bar(range(16), probabilities)
        axes[1, column].set_title(f"灰度分布：H={pixel_entropy:.2f} 比特")
        axes[1, column].set_xlabel("灰度区间（0–15）")
        axes[1, column].set_ylabel("像素占比")
        axes[1, column].set_ylim(0, 1)
        print(f"{name}：图片编号 {row['dataset_index']}，16 级灰度像素数 {counts}，灰度熵 {pixel_entropy:.4f} bits")
    figure.tight_layout()
    figure.savefig("pixel_distributions.png", dpi=150)
    plt.close(figure)
    print("已保存 pixel_distributions.png")
