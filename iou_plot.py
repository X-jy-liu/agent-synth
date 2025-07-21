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

def plot_ious(ious, if_save=False, save_path="iou_plot.png"):
    max_idx = int(max(enumerate(ious), key=lambda x: x[1])[0])
    max_iou = ious[max_idx]

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(ious)), ious, marker='o', color='blue', label="IoU")
    plt.plot(max_idx, max_iou, marker='o', color='red', label=f"Max IoU: {max_iou:.2f}")
    plt.annotate(f"{max_iou:.2f}", (max_idx, max_iou),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    plt.title("IoU Over Agent Steps")
    plt.xlabel("Step")
    plt.ylabel("IoU Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xticks(range(len(ious)))  # Integer-only x-axis
    plt.legend()
    plt.tight_layout()
    if if_save:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    ious = load_ious()
    if ious:
        plot_ious(ious, if_save=True, save_path="iou_plot.png")
    else:
        print("⚠️ No IoU scores found in outputs/")
