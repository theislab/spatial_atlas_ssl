# import packages
import scanpy as sc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

adata_path = sys.argv[1]
outfolder = sys.argv[2]

# check if outfolder exists and if not create it
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# load data
adata = sc.read_h5ad(adata_path)

# set sns color palette to colorblind
sns.set_palette('colorblind')

# set seaborn style to whitegrid
sns.set_style('whitegrid')

# process data to get cell type composition of sections
plot_data = adata.obs.groupby(['section', 'class_label']).size().reset_index().pivot(columns='class_label', index='section', values=0)


# first we do the barplot
plot_data.reset_index().plot.bar(stacked=True, x='section', figsize=(20, 10))
plt.legend().set_visible(False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# add title
plt.title('Cell type composition of sections', fontsize=30)

# add axis labels
plt.xlabel('Sections', fontsize=25)
plt.ylabel('Number of cells', fontsize=25)

# add legend on the right side
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)

# save figure with title "barplot_celltype_dist.png"
plt.savefig(outfolder + '/barplot_celltype_dist.png', bbox_inches='tight')


# reset plt figure
plt.clf()


# now we do the heatmap
# set figure size
plt.figure(figsize=(len(plot_data)* 2, 10))

# plot heatmap
sns.heatmap(plot_data.transpose(), cmap='Blues', annot=True, fmt='g', annot_kws={"size": 15})

# make plot prettier by removing spines all around
sns.despine()

# make font larger and plot easier to read
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# add title
plt.title('Cell type composition of sections', fontsize=20)

# increase font size of colorbar labels
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=15)

# add axis labels
plt.xlabel('Sections', fontsize=15)
plt.ylabel('Cell Types', fontsize=15)

# save figure with title "heatmap_celltype_dist.png"
plt.savefig(outfolder + '/heatmap_celltype_dist.png', bbox_inches='tight')


# reset plt figure
plt.clf()

# plot unique cell type distribution
counts_unique = adata.obs.groupby(['section'])['class_label'].nunique().reset_index()



# class labels are categorical, convert to integer
#counts_unique['class_label'] = counts_unique['class_label'].astype(int)

# create histogram plot and place bars directly above x axis ticks
counts_unique.plot.hist(figsize=(21, 10), bins=np.arange(adata.obs.class_label.nunique()+ 2)-0.5)

# make plot prettier by removing spines
#sns.despine()

# hide legend
plt.legend().set_visible(False)

# make font larger and plot easier to read
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# add title
plt.title('Number of unique cell types per section', fontsize=30)

#increase x axis label size
plt.xlabel('Number of unique cell types', fontsize=25)
plt.ylabel('Number of sections', fontsize=25)

# set xlimits to make plot easier to read
plt.xlim(0, len(adata.obs.class_label.unique())+ 2)

# only display whole integer values on x axis
plt.gca().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
plt.gca().get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))

# show all ticks on x axis
plt.xticks(np.arange(0, len(adata.obs.class_label.unique()) + 2, 1.0))

# save figure with title "histogram_unique_celltype_dist.png"
plt.savefig(outfolder + '/histogram_unique_celltype_dist.png', bbox_inches='tight')
