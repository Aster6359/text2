import math
import random

import torch
import torch.nn.functional as F
from torchvision.datasets import Imagenette
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)

small_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
large_weights = ResNet50_Weights.IMAGENET1K_V2

data = Imagenette(root="data", split="val", size="160px", download=False)
indices = random.Random(42).sample(range(len(data)), 300)

categories = small_weights.meta["categories"]
assert categories == large_weights.meta["categories"]
category_index = {name: index for index, name in enumerate(categories)}
label_map = torch.tensor([category_index[names[0]] for names in data.classes])
true_ids = torch.tensor([label_map[data[index][1]].item() for index in indices])


@torch.inference_mode()
def predict(model, preprocess, positions):
    batches = []
    for start in range(0, len(positions), 16):
        batch_positions = positions[start:start + 16]
        images = torch.stack([
            preprocess(data[indices[position]][0])
            for position in batch_positions
        ])
        batches.append(model(images))
    return torch.cat(batches)


small_model = mobilenet_v3_small(weights=small_weights).eval()
small_logits = predict(small_model, small_weights.transforms(), list(range(300)))
log_probs = small_logits.log_softmax(dim=1)
probs = log_probs.exp()
confidence, small_ids = probs.max(dim=1)
entropy = -(probs * log_probs).sum(dim=1) / math.log(2)

entropy_choice = torch.argsort(entropy, descending=True, stable=True)[:60].tolist()
confidence_choice = torch.argsort(confidence, stable=True)[:60].tolist()
print(f"两种策略共同选中的图片：{len(set(entropy_choice) & set(confidence_choice))}/60")

review_positions = sorted(set(entropy_choice) | set(confidence_choice))
large_model = resnet50(weights=large_weights).eval()
large_logits = predict(large_model, large_weights.transforms(), review_positions)
review_ids = dict(zip(review_positions, large_logits.argmax(dim=1).tolist()))
large_probs = large_logits.softmax(dim=1)

baseline_correct = small_ids == true_ids
print(f"MobileNet 基准 Top-1：{baseline_correct.float().mean():.2%}")

for name, selected in [("按熵复核", entropy_choice), ("按置信度复核", confidence_choice)]:
    final_ids = small_ids.clone()
    final_ids[selected] = torch.tensor([review_ids[position] for position in selected])
    final_correct = final_ids == true_ids

    fixed = (~baseline_correct & final_correct).sum().item()
    broken = (baseline_correct & ~final_correct).sum().item()
    print(
        f"{name}：复核 {len(selected)} 张，Top-1 {final_correct.float().mean():.2%}，"
        f"纠正 {fixed} 张，改错 {broken} 张，净增 {fixed - broken:+d} 张"
    )

example_position = entropy_choice[0]
large_position = review_positions.index(example_position)
examples = [
    ("MobileNet", probs[example_position]),
    ("ResNet50", large_probs[large_position]),
]
print(f"\n概率分布示例：图片编号 {indices[example_position]}，真实类别 {categories[true_ids[example_position]]}")
for name, distribution in examples:
    values, category_ids = distribution.topk(5)
    print(f"{name}：")
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
    for name, distribution in examples:
        ordered = distribution.sort(descending=True).values
        bits = (-torch.special.xlogy(distribution, distribution).sum() / math.log(2)).item()
        axes[0].plot(range(1, len(ordered) + 1), ordered.cumsum(0).tolist(),
                     label=f"{name}：熵={bits:.2f} 比特")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("按概率从高到低累计的类别数（对数刻度）")
    axes[0].set_ylabel("累计概率")
    axes[0].set_title("同一张图片的模型预测分布")
    axes[0].legend()

    group_names = ["全部图片", "高熵复核组", "低置信度复核组"]
    selections = [list(range(len(indices))), entropy_choice, confidence_choice]
    error_rates = [(~baseline_correct[group]).float().mean().item() for group in selections]
    axes[1].bar(group_names, error_rates)
    axes[1].set_ylabel("MobileNet 错误率")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("ResNet50 复核前的错误率")
    figure.suptitle(f"示例图片编号：{indices[example_position]}")
    figure.tight_layout()
    figure.savefig("routing_distributions.png", dpi=150)
    plt.close(figure)
    print("已保存 routing_distributions.png")

    image, _ = data[indices[example_position]]
    histogram = image.convert("L").histogram()
    counts = [sum(histogram[start:start + 16]) for start in range(0, 256, 16)]
    total = sum(counts)
    pixel_probabilities = [count / total for count in counts]
    pixel_entropy = -sum(probability * math.log2(probability) for probability in pixel_probabilities if probability > 0)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"图片编号：{indices[example_position]}")
    axes[0].axis("off")
    axes[1].bar(range(16), pixel_probabilities)
    axes[1].set_title(f"灰度分布：H={pixel_entropy:.2f} 比特")
    axes[1].set_xlabel("灰度区间（0–15）")
    axes[1].set_ylabel("像素占比")
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig("routing_pixel_distribution.png", dpi=150)
    plt.close(figure)
    print(f"16 级灰度像素数：{counts}，灰度熵：{pixel_entropy:.4f} bits")
    print("已保存 routing_pixel_distribution.png")

generator = torch.Generator().manual_seed(123)
order = torch.randperm(len(true_ids), generator=generator)
calibration_indices = order[:150]
test_indices = order[150:]

test_logits = small_logits[test_indices]
test_labels = true_ids[test_indices]
test_predictions = test_logits.argmax(dim=1)
accuracy = (test_predictions == test_labels).float().mean().item()
cross_entropy = F.cross_entropy(test_logits, test_labels).item()

print("\n--- 挑战二：MobileNet 原始结果 ---")
print(f"调参集数量：{len(calibration_indices)}")
print(f"测试集数量：{len(test_indices)}")
print(f"测试集准确率：{accuracy:.2%}")
print(f"测试集平均交叉熵：{cross_entropy:.4f}")

calibration_logits = small_logits[calibration_indices]
calibration_labels = true_ids[calibration_indices]
temperatures = [value / 100 for value in range(50, 301)]
calibration_losses = [
    F.cross_entropy(calibration_logits / temperature, calibration_labels).item()
    for temperature in temperatures
]
best_index = min(range(len(temperatures)), key=calibration_losses.__getitem__)
best_temperature = temperatures[best_index]

scaled_test_logits = test_logits / best_temperature
scaled_predictions = scaled_test_logits.argmax(dim=1)
scaled_accuracy = (scaled_predictions == test_labels).float().mean().item()
scaled_loss = F.cross_entropy(scaled_test_logits, test_labels).item()
changed_predictions = (scaled_predictions != test_predictions).sum().item()

print("\n--- 挑战二：温度缩放 ---")
print("搜索范围：0.50 至 3.00，步长 0.01")
print(f"调参集最佳温度：{best_temperature:.2f}")
print(f"调参集最低平均交叉熵：{calibration_losses[best_index]:.4f}")
print(f"测试集调整后准确率：{scaled_accuracy:.2%}")
print(f"测试集调整后平均交叉熵：{scaled_loss:.4f}")
print(f"测试集预测类别变化数：{changed_predictions}")
if best_index in (0, len(temperatures) - 1):
    print("最佳温度位于搜索边界，可以考虑扩大搜索范围。")

original_losses = F.cross_entropy(test_logits, test_labels, reduction="none")
scaled_losses = F.cross_entropy(scaled_test_logits, test_labels, reduction="none")
loss_changes = scaled_losses - original_losses
original_probabilities = test_logits.softmax(dim=1)
scaled_probabilities = scaled_test_logits.softmax(dim=1)
sample_positions = torch.arange(len(test_labels))
original_true_probs = original_probabilities[sample_positions, test_labels]
scaled_true_probs = scaled_probabilities[sample_positions, test_labels]

tolerance = 1e-6
decreased = (loss_changes < -tolerance).sum().item()
increased = (loss_changes > tolerance).sum().item()
unchanged = len(test_labels) - decreased - increased
print("\n--- 测试集逐图交叉熵变化 ---")
print(f"降低：{decreased} 张，升高：{increased} 张，基本不变：{unchanged} 张")

examples = [
    ("交叉熵降低最多", loss_changes.argmin().item()),
    ("交叉熵升高最多", loss_changes.argmax().item()),
]
for title, position in examples:
    sample_index = indices[test_indices[position].item()]
    label = categories[test_labels[position].item()]
    prediction = categories[test_predictions[position].item()]
    print(f"\n{title}：图片编号 {sample_index}")
    print(f"真实类别：{label}，预测类别：{prediction}")
    print(f"真实类别概率：{original_true_probs[position]:.4f} → {scaled_true_probs[position]:.4f}")
    print(f"交叉熵：{original_losses[position]:.4f} → {scaled_losses[position]:.4f}")


def analyze_confidence(logits, labels, title):
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    losses = F.cross_entropy(logits, labels, reduction="none")

    print(f"\n--- {title} ---")
    print("置信度区间    图片数    平均置信度    实际准确率    平均交叉熵")

    for group in range(5):
        lower = group / 5
        upper = (group + 1) / 5

        if group == 4:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = mask.sum().item()
        if count == 0:
            print(f"{lower:.1f}–{upper:.1f}          0       —             —             —")
            continue

        mean_confidence = confidence[mask].mean().item()
        accuracy = (predictions[mask] == labels[mask]).float().mean().item()
        mean_loss = losses[mask].mean().item()

        print(
            f"{lower:.1f}–{upper:.1f}"
            f"       {count:3d}"
            f"       {mean_confidence:6.2%}"
            f"       {accuracy:6.2%}"
            f"         {mean_loss:.4f}"
        )


analyze_confidence(test_logits, test_labels, "调整前置信度分组")
analyze_confidence(scaled_test_logits, test_labels, "调整后置信度分组")
