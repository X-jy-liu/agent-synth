import os
import matplotlib.pyplot as plt

def load_ious(output_root="outputs"):
    steps = sorted(
        [d for d in os.listdir(output_root) if d.startswith("step_")],
        key=lambda x: int(x.split("_")[1])
    )

    ious = []
    for step in steps:
        iou_path = os.path.join(output_root, step, "iou.txt")
        if os.path.exists(iou_path):
            with open(iou_path, "r") as f:
                try:
                    iou = float(f.read().strip())
                    ious.append(iou)
                except ValueError:
                    print(f"Invalid IoU in {iou_path}")
    return ious

def plot_ious(ious):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(ious)), ious, marker='o', color='blue', label="IoU")
    plt.title("IoU Over Agent Steps")
    plt.xlabel("Step")
    plt.ylabel("IoU Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ious = load_ious()
    if ious:
        plot_ious(ious)
    else:
        print("⚠️ No IoU scores found in outputs/")
