"""Generate the report's overview figure. Optional dependency: matplotlib."""
import hashlib
import json
from pathlib import Path

from .report_visuals import paired_metrics, SIZES

REPORTS = Path(__file__).parent / 'reports'


def chart_data(records):
    return {change: {size: [100*m['macro_f1'] for m in paired_metrics(records, size, change)]
                     for size in SIZES} for change in ['filter', 'finetune', 'compress']}


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = {p.parent.name: {'test': json.loads(p.read_text())}
               for p in REPORTS.glob('*/final-test.json')}
    data = chart_data(records)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'emotion-comparisons'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.9), sharey=True)
    fig.patch.set_facecolor('#ffffff')
    styles = [('tiny', '#3979b6', 'o', '-'), ('base', '#515b67', 's', '--'),
              ('small', '#152f51', 'D', ':')]
    specs = [('filter', 'Filter training clips', ['All clips', 'Agreement only']),
             ('finetune', 'Retrain encoder', ['Frozen encoder', 'Fine-tuned']),
             ('compress', 'Compress weights', ['Float', '8-bit'])]
    for ax, (change, title, labels) in zip(axes, specs):
        ax.set_facecolor('#ffffff')
        ax.set_ylim(50, 77)
        ax.set_xlim(-.45, 1.6)
        ax.set_yticks(range(50, 76, 5))
        ax.grid(axis='y', color='#e1e6ec')
        ax.axhline(75, color='#9d2838', linewidth=1.3, linestyle='--')
        ax.text(-.4, 75.6, 'Provisional target: 75', color='#9d2838', fontsize=10)
        for size, color, marker, line in styles:
            y = data[change][size]
            ax.plot([0, 1], y, color=color, marker=marker, linestyle=line,
                    linewidth=2.2, markersize=7, label=size.capitalize())
            offset = -17 if size == 'tiny' else 10
            for x, value in enumerate(y):
                ax.annotate(f'{value:.2f}', (x, value), xytext=(0, offset),
                            textcoords='offset points', ha='center', color=color, fontsize=10)
            gain = y[1]-y[0]
            ax.text(1.18, y[1], f'{gain:+.2f}', va='center', fontsize=10,
                    color='#137743' if gain >= 0 else '#b62e40', fontweight='bold')
        ax.text(1.18, 69.2, 'Change\n(points)', fontsize=9, color='#515b67')
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=18)
        ax.set_xticks([0, 1], labels)
        ax.tick_params(axis='both', length=0, pad=9, labelcolor='#273548')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#ccd4dd')
    axes[0].set_ylabel('Macro F1 score (0–100) · higher is better', labelpad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.97, .97),
               ncol=3, frameon=False)
    fig.text(.05, .95, 'What actually improved?', fontsize=21, fontweight='bold', color='#152f51')
    fig.text(.05, .885, 'Same 1,341 human-labelled test clips • three separate before/after comparisons',
             fontsize=12, color='#515b67')
    fig.text(.05, .075, 'Green change = gain; red change = loss. Line style / marker identifies model size. All models remain below target.',
             fontsize=10, color='#273548')
    fig.text(.05, .035, 'Vertical axis starts at 50. Descriptive point estimates; uncertainty is not shown. 135 tied human labels excluded.',
             fontsize=10, color='#515b67')
    fig.subplots_adjust(left=.075, right=.975, top=.75, bottom=.20, wspace=.12)
    svg = REPORTS/'modification-overview.svg'
    fig.savefig(svg, metadata={'Date': None}, facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(REPORTS/'modification-overview.png', dpi=140, facecolor='white')
    plt.close(fig)
    evidence = {'data': data, 'svg_sha256': hashlib.sha256(svg.read_bytes()).hexdigest()}
    (REPORTS/'modification-overview.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(svg)


if __name__ == '__main__':
    build()
