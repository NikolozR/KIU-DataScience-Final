import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_target_distribution(df, target_col='G3', save_path=None):
    """
    Plots the distribution of the target variable using histogram with KDE.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, bins=20, color='steelblue')
    plt.title(f"Distribution of {target_col} Grades", fontsize=14, fontweight='bold')
    plt.xlabel("Grade", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_correlation_heatmap(df, save_path=None, top_n=15):
    """
    Plots correlation heatmap for numeric columns.
    """
    plt.figure(figsize=(14, 12))
    numeric_df = df.select_dtypes(include=['number'])
    
    corr_matrix = numeric_df.corr()
    
    # Select top N most correlated with target
    if 'G3' in corr_matrix.columns and top_n:
        top_features = corr_matrix['G3'].abs().nlargest(top_n).index
        corr_matrix = corr_matrix.loc[top_features, top_features]
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, annot_kws={"size": 8}, center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap (Top Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_box_comparison(df, x_col, y_col, save_path=None):
    """
    Plots a boxplot to compare distributions across categories.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_col, y=y_col, data=df, palette='Set2')
    plt.title(f"{y_col} Distribution by {x_col}", fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_scatter_trend(df, x_col, y_col, save_path=None):
    """
    Plots a scatter plot with regression trendline.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha':0.5}, 
                line_kws={'color':'red', 'linewidth':2})
    plt.title(f"Relationship between {x_col} and {y_col}", fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_count_chart(df, col, save_path=None):
    """
    Plots a count plot (bar chart) for categorical variables.
    """
    plt.figure(figsize=(10, 6))
    order = df[col].value_counts().index
    sns.countplot(x=col, data=df, order=order, palette='viridis')
    plt.title(f"Count Distribution of {col}", fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_violin_comparison(df, x_col, y_col, save_path=None):
    """
    Plots a violin plot to show distribution and density across categories.
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=x_col, y=y_col, data=df, palette='muted', inner='box')
    plt.title(f"{y_col} Distribution by {x_col} (Violin Plot)", fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_pairplot(df, features, save_path=None):
    """
    Creates a pairplot for multivariate analysis of selected features.
    """
    if 'G3' not in features and 'G3' in df.columns:
        features = features + ['G3']
    
    pairplot = sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha':0.6})
    pairplot.fig.suptitle("Pairplot of Key Features", y=1.02, fontsize=14, fontweight='bold')
    
    if save_path:
        pairplot.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()



def plot_feature_importance(feature_names, importances, top_n=15, save_path=None):
    """
    Plots feature importance from tree-based models.
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='rocket')
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_confusion_matrix(cm, labels=['Failed', 'Passed'], save_path=None):
    """
    Plots a confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.8})
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def main():
    """
    Generates all EDA visualizations and saves them to reports/figures/.
    Uses student_for_viz.csv which has categorical columns intact.
    """
    df = pd.read_csv('data/processed/student_for_viz.csv')
    
    plot_target_distribution(df, save_path='reports/figures/01_target_distribution.png')
    plot_correlation_heatmap(df, save_path='reports/figures/02_correlation_heatmap.png')
    
    plot_box_comparison(df, 'subject', 'G3', save_path='reports/figures/03_boxplot_subject_g3.png')
    plot_scatter_trend(df, 'G1', 'G3', save_path='reports/figures/04_scatter_g1_g3.png')
    plot_count_chart(df, 'subject', save_path='reports/figures/05_countplot_subject.png')
    plot_violin_comparison(df, 'subject', 'G3', save_path='reports/figures/06_violin_subject_g3.png')
    plot_pairplot(df, ['G1', 'G2', 'studytime', 'failures'], 
                  save_path='reports/figures/07_pairplot.png')

if __name__ == "__main__":
    main()