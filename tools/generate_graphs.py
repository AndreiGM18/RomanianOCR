"""
Generate graphs comparing model performance.

Usage:
    python tools/generate_graphs.py
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metrics(metrics_file: Path) -> dict:
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_overall_accuracy(metrics: dict, output_dir: Path):
    models = list(metrics.keys())
    exact_acc = [metrics[m]['exact_accuracy'] * 100 for m in models]
    char_acc = [metrics[m]['character_accuracy'] * 100 for m in models]
    word_acc = [metrics[m]['word_accuracy'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, exact_acc, width, label='Exact Match', color='#2ecc71')
    bars2 = ax.bar(x, char_acc, width, label='Character', color='#3498db')
    bars3 = ax.bar(x + width, word_acc, width, label='Word', color='#e74c3c')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Generated {output_dir / 'overall_accuracy.png'}")


def generate_diacritic_accuracy(metrics: dict, output_dir: Path):
    diacritics = ['ă', 'â', 'î', 'ș', 'ț']
    models = list(metrics.keys())

    x = np.arange(len(diacritics))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        if 'diacritic_accuracy' in metrics[model]:
            values = [metrics[model]['diacritic_accuracy'].get(d, {}).get('accuracy', 0) * 100 for d in diacritics]
            offset = (i - len(models)//2) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Diacritic-Specific Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diacritics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'diacritic_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Generated {output_dir / 'diacritic_accuracy.png'}")


def generate_error_rate(metrics: dict, output_dir: Path):
    models = list(metrics.keys())
    cer = [metrics[m]['character_error_rate'] * 100 for m in models]
    wer = [metrics[m]['word_error_rate'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cer, width, label='Character Error Rate (CER)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, wer, width, label='Word Error Rate (WER)', color='#3498db')

    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Rate Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Generated {output_dir / 'error_rate.png'}")


def generate_summary_table(metrics: dict, output_dir: Path):
    models = list(metrics.keys())

    # Prepare table data
    rows = ['Exact Match', 'Character Acc', 'Character Error', 'Word Acc', 'Word Error', 'Diacritic Avg']
    table_data = []

    for model in models:
        m = metrics[model]
        diacritic_avg = 0
        if 'diacritic_accuracy' in m:
            acc_values = [v['accuracy'] for v in m['diacritic_accuracy'].values() if isinstance(v, dict)]
            if acc_values:
                diacritic_avg = np.mean(acc_values) * 100

        col_data = [
            f"{m['exact_accuracy']*100:.1f}%",
            f"{m['character_accuracy']*100:.2f}%",
            f"{m['character_error_rate']*100:.2f}%",
            f"{m['word_accuracy']*100:.1f}%",
            f"{m['word_error_rate']*100:.1f}%",
            f"{diacritic_avg:.1f}%"
        ]
        table_data.append(col_data)

    # Transpose for table format
    table_data = list(map(list, zip(*table_data)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data, rowLabels=rows, colLabels=models,
                    cellLoc='center', loc='center',
                    colWidths=[0.3] * len(models))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(models)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color row labels
    for i in range(len(rows)):
        table[(i+1, -1)].set_facecolor('#95a5a6')
        table[(i+1, -1)].set_text_props(weight='bold', color='white')

    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Generated {output_dir / 'summary_table.png'}")


def main():
    baselines_dir = Path('baselines')
    output_dir = baselines_dir / 'graphs'
    output_dir.mkdir(exist_ok=True)

    # Load all metrics files
    metrics_files = {
        'EN PP-OCRv3': baselines_dir / 'en_ppocrv3_val_predictions.metrics.json',
        'Latin PP-OCRv5': baselines_dir / 'ppocrv5_latin_val_predictions.metrics.json',
        'Romanian PP-OCRv3': baselines_dir / 'romanian_ppocrv3_final_val_predictions.metrics.json',
    }

    metrics = {}
    for name, path in metrics_files.items():
        if not path.exists():
            logger.warning(f"Metrics file not found: {path}")
            continue
        metrics[name] = load_metrics(path)
        logger.info(f"Loaded metrics from {path}")

    if not metrics:
        logger.error("No metrics files found")
        return

    logger.info("Generating graphs...")
    generate_overall_accuracy(metrics, output_dir)
    generate_diacritic_accuracy(metrics, output_dir)
    generate_error_rate(metrics, output_dir)
    generate_summary_table(metrics, output_dir)

    logger.info(f"\nDone: {output_dir}/")


if __name__ == '__main__':
    main()
