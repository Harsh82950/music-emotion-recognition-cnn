import os
import matplotlib.pyplot as plt


def save_metrics(train_losses, val_losses, epochs):
    os.makedirs("results/metrics", exist_ok=True)

    with open("results/metrics/final_metrics.txt", "w") as f:
        f.write("Music Emotion Recognition - Results\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Final Train MSE: {train_losses[-1]:.4f}\n")
        f.write(f"Final Val MSE: {val_losses[-1]:.4f}\n")


def save_plots(train_losses, val_losses):
    os.makedirs("results/plots", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/plots/loss_curve.png")
    plt.close()
