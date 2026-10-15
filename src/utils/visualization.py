import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import seaborn as sns



def visualize_batch(images, labels, preds, savedir, savename =  'batch_visualization'):
    """
    Visualize a batch of images with their true labels and predicted labels."""
 
    batch_size = images.size(0)
    # build a grid according to batch size
    n_cols = min(batch_size, 8)
    n_rows = (batch_size + n_cols - 1) // n_cols
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for i in range(batch_size):
        # print image values

        row = i // n_cols
        col = i % n_cols
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        # imagenet renomalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'True: {labels[i]}\nPred: {preds[i]}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'{savename}'))
    plt.close()




def save_metrics_and_plot(
        name,                 # es: "teacher", "student", "student_kd"
        train_losses,
        val_losses,
        val_f1s,
        results_dir
    ):

    os.makedirs(results_dir, exist_ok=True)

    epochs_range = range(len(train_losses))

    # -------------------- CSV --------------------
    csv_path = os.path.join(results_dir, f"{name}_plots.csv")

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in epochs_range:
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_losses[epoch],
                'val_loss': val_losses[epoch],
                'val_f1': val_f1s[epoch]
            })

    print(f"Salvato CSV: {csv_path}")

    # -------------------- PLOTTING --------------------
    plt.figure(figsize=(14, 5))

    # LOSS
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=train_losses, label="Train Loss", linewidth=2.5)
    sns.lineplot(x=epochs_range, y=val_losses, label="Val Loss", linewidth=2.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{name.replace('_', ' ').title()} - Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # F1
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=val_f1s, label="Val F1", linewidth=2.5)
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title(f"{name.replace('_', ' ').title()} - F1 Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    img_path = os.path.join(results_dir, f"{name}_plots.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Salvato plot: {img_path}\n")



