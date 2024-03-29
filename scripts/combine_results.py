import glob
import sys
import time

import pandas as pd
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
#sns.set_theme(style="whitegrid")
#sns.set_context("paper", font_scale=1.5)


inputfolder = sys.argv[1]
outputfile = sys.argv[2]

#print(os.getcwd())
#print(sys.argv)
matching_files = glob.glob(inputfolder + "/summary_*.csv")

#print(matching_files)

# read all files and concatenate them pandas, add file basename as column
df = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in matching_files])

# plot best val loss using seaborn
with PdfPages(outputfile) as pdf:
    ax = sns.barplot(x="file", y="best_val_loss", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.tight_layout()

    # auto set y limit based on data
    ax.set_ylim(df["best_val_loss"].min() / 1.1, df["best_val_loss"].max() * 1.1)

    # add labels to bars in white color
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, -10), textcoords='offset points', color='white')

    # add title and axis labels
    ax.set_ylabel("Best validation loss")
    ax.set_xlabel("Model")
    pdf.savefig(ax.figure)
    ax.figure.clear()

    # find metric cloumn
    metric_column = [col for col in df.columns if 'best_val' in col and 'loss' not in col][0]

    # plot highest r2 using seaborn
    ax = sns.barplot(x="file", y=metric_column, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.tight_layout()

    # auto set y limit based on data
    ax.set_ylim(df[metric_column].min() / 1.1, df[metric_column].max() * 1.1)

    # add labels to bars in white color
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, -10), textcoords='offset points', color='white')

    # add title and axis labels
    if "R2" in metric_column:
        ax.set_ylabel("Best R2 score")
    elif "Acc" in metric_column:
        ax.set_ylabel("Best accuracy")

    ax.set_xlabel("Model")
    pdf.savefig(ax.figure)
    ax.figure.clear()


