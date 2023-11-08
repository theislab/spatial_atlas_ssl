# import packages
import pandas as pd
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

# create example data for plotting with 50 sections with names and 4 cell types
#plot_data = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=['section', 'cell_type_2', 'cell_type_3', 'cell_type_4'])

# melt table to get cell type counts per section
#plot_data = pd.melt(plot_data.reset_index(), id_vars=['section'], value_vars=['cell_type_2', 'cell_type_3', 'cell_type_4'])


# concat plot data four times
#plot_data = pd.concat([plot_data, plot_data, plot_data, plot_data], axis=1)


# first we do the barplot
plot_data.reset_index().plot.bar(stacked=True, x='section', figsize=(16, 10))
plt.legend().set_visible(False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# add title
plt.title('Cell Type Composition of Slices', fontsize=30)

# add axis labels
plt.xlabel('Slice', fontsize=25)
plt.ylabel('Number of Cells', fontsize=25)

# set x labels to 1 and 2
# labels one to number of sections
labels = np.arange(1, len(plot_data)+ 1, 1)
xticks = np.arange(0, len(plot_data), 1)

#remove x labels
plt.xticks([])


plt.xticks(labels=labels, ticks=xticks, fontsize=10)

# rotate x axis labels
plt.xticks(rotation=0)

# make space between bars smaller
plt.subplots_adjust(wspace=0.1)

# make plot prettier by removing spines all around
sns.despine()

# add legend on the right side
#plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)

# hide legends
plt.legend().set_visible(False)



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
plt.title('Cell Type Composition across Slices', fontsize=20)

# increase font size of colorbar labels
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=15)

# add axis labels
plt.xlabel('Slices', fontsize=15)
plt.ylabel('Cell Types', fontsize=15)

# hide legend
plt.legend().set_visible(False)

# set x labels to 1 to 59
#plt.xticks(np.arange(0, len(plot_data)+ 1, 1.0))



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

