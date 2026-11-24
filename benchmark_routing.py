import random
import statistics
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import Imagenette
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)


SAMPLE_COUNT = 300
CALIBRATION_COUNT = 150
REVIEW_FRACTIONS = (0.0, 0.20, 0.40, 0.60, 1.0)
BATCH_SIZE = 16
REPEATS = 3


@torch.inference_mode()
def predict(model, transform, images):
    outputs = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = torch.stack([transform(image) for image in images[start:start + BATCH_SIZE]])
        outputs.append(model(batch))
    return torch.cat(outputs)


def main():
    root = Path(__file__).resolve().parent
    data = Imagenette(root=root / "data", split="val", size="160px", download=False)
    indices = random.Random(42).sample(range(len(data)), SAMPLE_COUNT)

    generator = torch.Generator().manual_seed(123)
    order = torch.randperm(SAMPLE_COUNT, generator=generator).tolist()
    calibration_indices = [indices[position] for position in order[:CALIBRATION_COUNT]]
    test_indices = [indices[position] for position in order[CALIBRATION_COUNT:]]

    small_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    large_weights = ResNet50_Weights.IMAGENET1K_V2
    categories = small_weights.meta["categories"]
    if categories != large_weights.meta["categories"]:
        raise ValueError("两个模型的类别顺序不一致")
    category_index = {name: index for index, name in enumerate(categories)}
    label_map = [category_index[names[0]] for names in data.classes]

    calibration_images = [data[index][0] for index in calibration_indices]
    test_samples = [data[index] for index in test_indices]
    test_images = [image for image, label in test_samples]
    test_labels = torch.tensor([label_map[label] for image, label in test_samples])

    small_model = mobilenet_v3_small(weights=small_weights).eval()
    large_model = resnet50(weights=large_weights).eval()
    small_transform = small_weights.transforms()
    large_transform = large_weights.transforms()

    calibration_logits = predict(small_model, small_transform, calibration_images)
    calibration_confidence = calibration_logits.softmax(dim=1).max(dim=1).values
    sorted_confidence = calibration_confidence.sort().values

    def small_only():
        logits = predict(small_model, small_transform, test_images)
        return logits.argmax(dim=1), 0

    def large_only():
        logits = predict(large_model, large_transform, test_images)
        return logits.argmax(dim=1), len(test_images)

    def routed(threshold):
        logits = predict(small_model, small_transform, test_images)
        confidence, predictions = logits.softmax(dim=1).max(dim=1)
        positions = (confidence <= threshold).nonzero(as_tuple=True)[0].tolist()
        if positions:
            review_images = [test_images[position] for position in positions]
            review_logits = predict(large_model, large_transform, review_images)
            predictions[positions] = review_logits.argmax(dim=1)
        return predictions, len(positions)

    def benchmark(name, method):
        method()
        durations = []
        predictions = None
        review_count = 0
        for _ in range(REPEATS):
            start = time.perf_counter()
            predictions, review_count = method()
            durations.append(time.perf_counter() - start)
        median = statistics.median(durations)
        accuracy = (predictions == test_labels).float().mean().item()
        print(f"\n{name}")
        print(f"准确率：{accuracy:.2%}")
        print(f"大模型处理：{review_count}/{len(test_images)} 张")
        print("每次耗时：" + "、".join(f"{duration:.3f} 秒" for duration in durations))
        print(f"中位耗时：{median:.3f} 秒")
        print(f"吞吐量：{len(test_images) / median:.2f} 张/秒")
        return review_count / len(test_images), accuracy, len(test_images) / median

    print(f"CPU 线程数：{torch.get_num_threads()}，批大小：{BATCH_SIZE}，重复次数：{REPEATS}")
    print(f"调参集：{len(calibration_images)} 张，测试集：{len(test_images)} 张")
    print("计时包含预处理和推理，图片解码与模型加载在计时前完成。")
    print("中间档位的测试集复核比例由调参集阈值决定。")

    results = []
    for fraction in REVIEW_FRACTIONS:
        if fraction == 0:
            result = benchmark("0%：MobileNet 全量", small_only)
        elif fraction == 1:
            result = benchmark("100%：ResNet50 全量", large_only)
        else:
            target_count = round(len(calibration_indices) * fraction)
            threshold = sorted_confidence[target_count - 1].item()
            calibration_count = (calibration_confidence <= threshold).sum().item()
            print(f"\n目标 {fraction:.0%}：阈值 {threshold:.6f}，调参集复核 {calibration_count} 张")
            result = benchmark(f"目标复核 {fraction:.0%}", lambda: routed(threshold))
        results.append(result)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    review_rates = [result[0] * 100 for result in results]
    accuracies = [result[1] * 100 for result in results]
    throughputs = [result[2] for result in results]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(review_rates, accuracies, marker="o")
    axes[0].set_xlabel("测试集实际大模型处理比例（%）")
    axes[0].set_ylabel("准确率（%）")
    axes[0].set_title("准确率与大模型处理比例")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(review_rates, throughputs, marker="o", color="tab:orange")
    axes[1].set_xlabel("测试集实际大模型处理比例（%）")
    axes[1].set_ylabel("吞吐量（张/秒）")
    axes[1].set_title("吞吐量与大模型处理比例")
    axes[1].grid(True, alpha=0.3)
    for index, fraction in enumerate(REVIEW_FRACTIONS):
        for axis, values in ((axes[0], accuracies), (axes[1], throughputs)):
            axis.annotate(f"目标 {fraction:.0%}", (review_rates[index], values[index]),
                          xytext=(5, 6), textcoords="offset points", fontsize=8)
    figure.suptitle("不同复核比例的准确率与吞吐量")
    figure.tight_layout()
    output = root / "review_tradeoff.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"\n已保存图表：{output}")


if __name__ == "__main__":
    main()
