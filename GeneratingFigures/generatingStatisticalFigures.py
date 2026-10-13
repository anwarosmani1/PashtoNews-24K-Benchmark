import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
# FORCE UTF-8 ENCODING FOR WINDOWS CONSOLE
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_dataset(filepath):
    """Load JSONL dataset"""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def analyze_dataset(articles):
    """Generate comprehensive statistics"""
    df = pd.DataFrame(articles)
    
    # Calculate lengths first to avoid repeated computation
    df['title_len'] = df['title'].str.len()
    df['text_len'] = df['text'].str.len()
    
    stats = {
        'total_articles': len(articles),
        'sources': df['source'].nunique(),
        'categories': df['category'].nunique(),
        'articles_by_source': df['source'].value_counts().to_dict(),
        'articles_by_category': df['category'].value_counts().to_dict(),
        'title_length': {
            'mean': df['title_len'].mean(),
            'median': df['title_len'].median(),  # <-- FIXED: added .len() logic via precomputed column
            'min': df['title_len'].min(),
            'max': df['title_len'].max(),
            'std': df['title_len'].std()
        },
        'text_length': {
            'mean': df['text_len'].mean(),
            'median': df['text_len'].median(),
            'min': df['text_len'].min(),
            'max': df['text_len'].max(),
            'std': df['text_len'].std()
        }
    }
    
    # Source-Category distribution
    stats['source_category_dist'] = pd.crosstab(df['source'], df['category'])
    
    return stats, df

def create_visualizations(df, stats, output_dir='figures'):
    """Generate publication-quality figures"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Figure 1: Articles by Category
    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts = df['category'].value_counts()
    colors = sns.color_palette("viridis", len(category_counts))
    bars = ax.bar(range(len(category_counts)), category_counts.values, color=colors)
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title('Distribution of Articles by Category', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, category_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(category_counts.values)*0.01), 
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/articles_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Articles by Source
    fig, ax = plt.subplots(figsize=(10, 6))
    source_counts = df['source'].value_counts()
    colors = sns.color_palette("coolwarm", len(source_counts))
    bars = ax.bar(range(len(source_counts)), source_counts.values, color=colors)
    ax.set_xticks(range(len(source_counts)))
    ax.set_xticklabels(source_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title('Distribution of Articles by Source', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, source_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(source_counts.values)*0.01), 
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/articles_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Source-Category Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    source_category_matrix = pd.crosstab(df['source'], df['category'])
    sns.heatmap(source_category_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                ax=ax, linewidths=.5, cbar_kws={'label': 'Number of Articles'})
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Source', fontsize=12)
    ax.set_title('Distribution of Articles Across Sources and Categories', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/source_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Article Length Distribution (Text)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall text length histogram
    axes[0, 0].hist(df['text_len'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(stats['text_length']['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {stats["text_length"]["mean"]:.0f}')
    axes[0, 0].axvline(stats['text_length']['median'], color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {stats["text_length"]["median"]:.0f}')
    axes[0, 0].set_xlabel('Text Length (characters)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Overall Text Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    # Text length by category
    text_len_by_cat = df.groupby('category')['text_len'].median()
    colors = sns.color_palette("viridis", len(text_len_by_cat))
    axes[0, 1].bar(range(len(text_len_by_cat)), text_len_by_cat.values, color=colors)
    axes[0, 1].set_xticks(range(len(text_len_by_cat)))
    axes[0, 1].set_xticklabels(text_len_by_cat.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Median Text Length (characters)', fontsize=11)
    axes[0, 1].set_title('Median Text Length by Category', fontsize=12, fontweight='bold')
    
    # Title length histogram
    axes[1, 0].hist(df['title_len'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(stats['title_length']['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {stats["title_length"]["mean"]:.0f}')
    axes[1, 0].set_xlabel('Title Length (characters)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Title Length Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    
    # Text length by source
    text_len_by_source = df.groupby('source')['text_len'].median()
    colors = sns.color_palette("coolwarm", len(text_len_by_source))
    axes[1, 1].bar(range(len(text_len_by_source)), text_len_by_source.values, color=colors)
    axes[1, 1].set_xticks(range(len(text_len_by_source)))
    axes[1, 1].set_xticklabels(text_len_by_source.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Median Text Length (characters)', fontsize=11)
    axes[1, 1].set_title('Median Text Length by Source', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Pie chart for category distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    category_counts = df['category'].value_counts()
    colors = sns.color_palette("viridis", len(category_counts))
    wedges, texts, autotexts = ax.pie(category_counts.values, labels=category_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90,
                                       textprops={'fontsize': 11})
    ax.set_title('Category Distribution (Percentage)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All figures saved to '{output_dir}/' directory")

def generate_tables(stats, output_dir='tables'):
    """Generate LaTeX tables for the paper"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Table 1: Dataset Statistics
    table1 = r"""
\begin{table}[h]
\centering
\caption{Dataset Statistics Overview}
\label{tab:dataset_stats}
\begin{tabular}{lr}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Total Articles & """ + f"{stats['total_articles']}" + r""" \\
Number of Sources & """ + f"{stats['sources']}" + r""" \\
Number of Categories & """ + f"{stats['categories']}" + r""" \\
\hline
\multicolumn{2}{l}{\textbf{Text Length Statistics (characters)}} \\
Mean & """ + f"{stats['text_length']['mean']:.2f}" + r""" \\
Median & """ + f"{stats['text_length']['median']:.0f}" + r""" \\
Standard Deviation & """ + f"{stats['text_length']['std']:.2f}" + r""" \\
Minimum & """ + f"{stats['text_length']['min']}" + r""" \\
Maximum & """ + f"{stats['text_length']['max']}" + r""" \\
\hline
\multicolumn{2}{l}{\textbf{Title Length Statistics (characters)}} \\
Mean & """ + f"{stats['title_length']['mean']:.2f}" + r""" \\
Median & """ + f"{stats['title_length']['median']:.0f}" + r""" \\
Standard Deviation & """ + f"{stats['title_length']['std']:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/table1_dataset_stats.tex', 'w', encoding='utf-8') as f:
        f.write(table1)
    
    # Table 2: Articles by Category
    table2 = r"""
\begin{table}[h]
\centering
\caption{Distribution of Articles by Category}
\label{tab:category_dist}
\begin{tabular}{lr}
\hline
\textbf{Category} & \textbf{Number of Articles} \\
\hline
"""
    for cat, count in stats['articles_by_category'].items():
        table2 += f"{cat.replace('_', ' ').title()} & {count} \\\\\n"
    
    table2 += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/table2_category_dist.tex', 'w', encoding='utf-8') as f:
        f.write(table2)
    
    # Table 3: Articles by Source
    table3 = r"""
\begin{table}[h]
\centering
\caption{Distribution of Articles by Source}
\label{tab:source_dist}
\begin{tabular}{lr}
\hline
\textbf{Source} & \textbf{Number of Articles} \\
\hline
"""
    for source, count in stats['articles_by_source'].items():
        table3 += f"{source.title()} & {count} \\\\\n"
    
    table3 += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/table3_source_dist.tex', 'w', encoding='utf-8') as f:
        f.write(table3)
    
    # Table 4: Source-Category Cross-tabulation
    table4 = r"""
\begin{table}[h]
\centering
\caption{Cross-tabulation of Articles by Source and Category}
\label{tab:source_category}
\begin{tabular}{l""" + 'r' * len(stats['articles_by_category']) + r"""}
\hline
\textbf{Source} & """ + ' & '.join([cat.replace('_', ' ').title() for cat in stats['articles_by_category'].keys()]) + r""" \\
\hline
"""
    for source in stats['source_category_dist'].index:
        row = f"{source.title()}"
        for cat in stats['source_category_dist'].columns:
            row += f" & {stats['source_category_dist'].loc[source, cat]}"
        row += " \\\\\n"
        table4 += row
    
    table4 += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/table4_source_category.tex', 'w', encoding='utf-8') as f:
        f.write(table4)
    
    print(f"✓ All LaTeX tables saved to '{output_dir}/' directory")

def print_summary(stats):
    """Print a console summary"""
    print("\n" + "="*70)
    print("PASHTO NEWS DATASET - STATISTICAL SUMMARY")
    print("="*70)
    print(f"\nTotal Articles: {stats['total_articles']}")
    print(f"Number of Sources: {stats['sources']}")
    print(f"Number of Categories: {stats['categories']}")
    
    print("\n" + "-"*70)
    print("ARTICLES BY CATEGORY:")
    print("-"*70)
    for cat, count in sorted(stats['articles_by_category'].items()):
        percentage = (count / stats['total_articles']) * 100
        print(f"  {cat.replace('_', ' ').title():30s} {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "-"*70)
    print("ARTICLES BY SOURCE:")
    print("-"*70)
    for source, count in sorted(stats['articles_by_source'].items()):
        percentage = (count / stats['total_articles']) * 100
        print(f"  {source.title():30s} {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "-"*70)
    print("TEXT LENGTH STATISTICS (characters):")
    print("-"*70)
    print(f"  Mean:     {stats['text_length']['mean']:10.2f}")
    print(f"  Median:   {stats['text_length']['median']:10.0f}")
    print(f"  Std Dev:  {stats['text_length']['std']:10.2f}")
    print(f"  Min:      {stats['text_length']['min']:10d}")
    print(f"  Max:      {stats['text_length']['max']:10d}")
    
    print("\n" + "-"*70)
    print("TITLE LENGTH STATISTICS (characters):")
    print("-"*70)
    print(f"  Mean:     {stats['title_length']['mean']:10.2f}")
    print(f"  Median:   {stats['title_length']['median']:10.0f}")
    print(f"  Std Dev:  {stats['title_length']['std']:10.2f}")
    print(f"  Min:      {stats['title_length']['min']:10d}")
    print(f"  Max:      {stats['title_length']['max']:10d}")
    print("="*70 + "\n")

# Main execution
if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    #input file (Dataset)
    articles = load_dataset('Pashto-news-24K.jsonl')  
    
    # Analyze dataset
    print("Analyzing dataset...")
    stats, df = analyze_dataset(articles)
    
    # Print summary to console
    print_summary(stats)
    
    # Generate visualizations
    print("Generating visualizations...")
    create_visualizations(df, stats, output_dir='data_statistics_figures')
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_tables(stats, output_dir='tables')
    
    print("\n✓ Analysis complete! Check the 'figures' and 'tables' directories.")