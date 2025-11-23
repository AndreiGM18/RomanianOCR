import re
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse_train_log(log_path):
    train_metrics = []
    val_metrics = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Training metrics
            if 'epoch:' in line and 'global_step:' in line and 'acc:' in line:
                match = re.search(
                    r'epoch: \[(\d+)/\d+\], global_step: (\d+), lr: ([\d.]+), acc: ([\d.]+).*?loss: ([\d.]+)',
                    line
                )
                if match:
                    epoch, step, lr, acc, loss = match.groups()
                    train_metrics.append({
                        'epoch': int(epoch),
                        'step': int(step),
                        'lr': float(lr),
                        'acc': float(acc),
                        'loss': float(loss)
                    })

            # Validation metrics
            if 'best metric' in line:
                match = re.search(
                    r'best metric, acc: ([\d.]+).*?norm_edit_dis: ([\d.]+).*?best_epoch: (\d+)',
                    line
                )
                if match:
                    acc, norm_edit_dis, best_epoch = match.groups()
                    val_metrics.append({
                        'acc': float(acc),
                        'norm_edit_dis': float(norm_edit_dis),
                        'best_epoch': int(best_epoch)
                    })

    return train_metrics, val_metrics


def aggregate_by_epoch(train_metrics):
    epochs = {}
    for metric in train_metrics:
        epoch = metric['epoch']
        if epoch not in epochs:
            epochs[epoch] = {
                'acc': [],
                'loss': [],
                'lr': []
            }
        epochs[epoch]['acc'].append(metric['acc'])
        epochs[epoch]['loss'].append(metric['loss'])
        epochs[epoch]['lr'].append(metric['lr'])

    # Calculate averages
    aggregated = []
    for epoch in sorted(epochs.keys()):
        aggregated.append({
            'epoch': epoch,
            'avg_acc': sum(epochs[epoch]['acc']) / len(epochs[epoch]['acc']),
            'avg_loss': sum(epochs[epoch]['loss']) / len(epochs[epoch]['loss']),
            'avg_lr': sum(epochs[epoch]['lr']) / len(epochs[epoch]['lr'])
        })

    return aggregated


def plot_metrics(aggregated_train, val_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Training accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = [m['epoch'] for m in aggregated_train]
    acc = [m['avg_acc'] for m in aggregated_train]
    ax.plot(epochs, acc, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_accuracy.png', dpi=150)
    plt.close()

    # Training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    loss = [m['avg_loss'] for m in aggregated_train]
    ax.plot(epochs, loss, 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150)
    plt.close()

    # Learning rate
    fig, ax = plt.subplots(figsize=(10, 6))
    lr = [m['avg_lr'] for m in aggregated_train]
    ax.plot(epochs, lr, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate.png', dpi=150)
    plt.close()

    # Validation accuracy
    if val_metrics:
        # Group by unique entries (remove duplicates)
        unique_val = []
        seen = set()
        for m in val_metrics:
            key = (m['acc'], m['norm_edit_dis'])
            if key not in seen:
                seen.add(key)
                unique_val.append(m)

        fig, ax = plt.subplots(figsize=(10, 6))
        val_acc = [m['acc'] for m in unique_val]
        val_x = list(range(1, len(val_acc) + 1))
        ax.plot(val_x, val_acc, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Validation Checkpoint')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_accuracy.png', dpi=150)
        plt.close()

    print(f"Graphs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from PaddleOCR log')
    parser.add_argument('--log', type=str,
                       default='output/romanian_ppocrv3/train.log',
                       help='Path to training log file')
    parser.add_argument('--output-dir', type=str,
                       default='baselines/graphs',
                       help='Output directory for graphs')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Save metrics to JSON file')

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    print(f"Parsing {log_path}")
    train_metrics, val_metrics = parse_train_log(log_path)

    print(f"Found {len(train_metrics)} training entries")
    print(f"Found {len(val_metrics)} validation entries")

    if not train_metrics:
        print("No training metrics found")
        return

    print("\nAggregating by epoch...")
    aggregated = aggregate_by_epoch(train_metrics)

    print("Generating graphs...")
    plot_metrics(aggregated, val_metrics, args.output_dir)

    if args.save_metrics:
        output_dir = Path(args.output_dir)
        metrics_file = output_dir / 'training_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'train': aggregated,
                'validation': val_metrics
            }, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
