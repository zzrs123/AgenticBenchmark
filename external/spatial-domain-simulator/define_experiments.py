import pandas as pd
import numpy as np
import os
import random
from itertools import permutations
import tqdm

from create_tissues import make_straight_stripes, make_dataset
from assign_counts import map_spottypes, assign_spots


def save_to_files(counts, coords, labels, savepath):
    counts.to_csv(f'{savepath}_UMImatrix.csv', index=False, header=False)
    coords.to_csv(f'{savepath}_coords.csv', index=False, header=False)
    labels.to_csv(f'{savepath}_labels.csv', index=False, header=False)


def generate_blank_experiment(adata, regioncolname, celltypes_df, shape, cell_assignment_seed=1):
    coords_df, labels_df = make_dataset(shape, n_bg=4000)
    labels_df = map_spottypes(labels_df, celltypes_df, 1)
    newadata = assign_spots(adata, labels_df, regioncolname, randomstate=29+cell_assignment_seed)
    newadata = newadata.to_memory()
    newcounts = newadata.to_df()
    return newcounts, coords_df, labels_df['label']


def modify_resolution(counts, coords, labels, gridlength=0.5):
    random.seed(1)
    gridcoords = (coords.copy()/gridlength).round()*gridlength
    assert (gridcoords.index == labels.index).all()
    counts.index = labels.index
    indexgroups = gridcoords.reset_index().groupby(['x', 'y'])['index'].apply(list)
    indexdict = {oldindex: newindex for newindex, sublist in enumerate(indexgroups.tolist()) for oldindex in sublist}
    gridcoords['newindex'] = gridcoords.index.map(indexdict)
    newgridcoords = gridcoords.groupby('newindex')[['x', 'y']].apply('mean')
    assert (newgridcoords == gridcoords.drop_duplicates()[['x', 'y']].sort_values(['x', 'y']).reset_index(drop=True)).all().all()
    newlabels = labels.to_frame().join(pd.Series(labels.index.map(indexdict), name='newindex'))
    newgridlabels = newlabels.groupby('newindex').apply(lambda x: x.mode().iloc[0])
    counts['newindex'] = labels.index.map(indexdict)
    newgridcounts = counts.groupby('newindex').apply('mean').round()
    return newgridcounts, newgridcoords, newgridlabels['label']


def modify_number_of_genes(counts, coords, labels, downsample_prop=1):
    random.seed(1)
    ngenes = int(downsample_prop*counts.shape[1])
    print(f'downsample to ngenes: {ngenes}')
    if downsample_prop==1:
        lesscounts = counts
    else:
        lesscounts = counts[random.sample(counts.columns.tolist(), ngenes)]
    return lesscounts, coords, labels
            

def modify_sparsity(counts, coords, labels, goalprop=0.84):
    if goalprop < 0.84:
        raise ValueError("goalprop must be >=0.84. ")
    n_counts = np.prod(counts.shape)
    nonzero_indices = np.nonzero(counts)
    nonzero_indices = pd.DataFrame({'cell': nonzero_indices[0], 'gene': nonzero_indices[1]})
    goal_n_zero = int(goalprop*n_counts)
    set_to_zero = goal_n_zero - (n_counts-len(nonzero_indices))
    indices_to_remove = nonzero_indices.sample(set_to_zero, random_state=1)
    newcounts = counts.copy()
    counts_array = newcounts.to_numpy()
    counts_array[indices_to_remove['cell'], indices_to_remove['gene']]=0
    print(f'old sparsity: {(counts==0).sum(axis=1).sum()/n_counts:.2f}')
    print(f'new sparsity: {(newcounts==0).sum(axis=1).sum()/n_counts:.2f}')
    return newcounts, coords, labels


def counts_perturbation_type1(adata, regioncolname, celltypes_df, shape, noise_proportion=0, cell_assignment_seed=1):
    coords_df, labels_df = make_dataset(shape, n_bg=4000)
    labels_df = map_spottypes(labels_df, celltypes_df, 1)
    newadata = assign_spots(adata, labels_df, regioncolname, randomstate=29+cell_assignment_seed)
    newadata = newadata.to_memory()
    newcounts = newadata.to_df().reset_index(drop=True)
    noise_df = map_spottypes(pd.DataFrame(['noise']*len(newcounts.index), index=newcounts.index, columns=['label']), celltypes_df, 1)
    noiseadata = assign_spots(adata, noise_df, regioncolname, randomstate=29+cell_assignment_seed)
    noiseadata = noiseadata.to_memory()
    noisecounts = noiseadata.to_df().reset_index(drop=True)
    totalcounts = (noise_proportion*noisecounts).add((1-noise_proportion)*newcounts)
    return totalcounts, coords_df, labels_df['label']


def counts_perturbation_type2(adata, regioncolname, celltypes_df, shape, noise_proportion=0, cell_assignment_seed=1):
    import warnings
    coords_df, labels_df = make_dataset(shape, n_bg=4000)
    random.seed(1)
    newlabels = labels_df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noiseindex = random.sample(newlabels.index.tolist(), int(noise_proportion*len(newlabels.index)))
        newlabels.loc[noiseindex] = 'noise'
    newlabels = map_spottypes(newlabels, celltypes_df, 1)
    newadata = assign_spots(adata, newlabels, regioncolname, randomstate=29+cell_assignment_seed)
    newcounts = newadata.to_memory().to_df()
    return newcounts, coords_df, labels_df['label']


def vary_thickness(adata, regioncolname, celltypes_df, reverse_assignments=False, divides=(30, 60), cell_assignment_seed=1):
    coords_df, labels_df = make_straight_stripes(n_bg=4000, divisions=divides)
    if reverse_assignments:
        labels_df[labels_df['label']==2] = 0
        labels_df[labels_df['label']==3] = 2
        labels_df[labels_df['label']==0] = 3
    labels_df = map_spottypes(labels_df, celltypes_df, 1)
    newadata = assign_spots(adata, labels_df, regioncolname, randomstate=29+cell_assignment_seed)
    newcounts = newadata.to_memory().to_df()
    return newcounts, coords_df, labels_df['label']


def vary_shapes(adata, regioncolname, celltypes_df, shape, permutation='1234'):
    n_labels = 4 if shape in ['stripes','blobs','varied'] else 3
    assert len(permutation) == n_labels
    coords_df, labels_df = make_dataset(shape, n_bg=4000)
    labels_df['label'] = labels_df['label'].map(dict(zip((1,2,3,4), permutation))).astype(int)
    labels_df = map_spottypes(labels_df, celltypes_df, 1)
    newadata = assign_spots(adata, labels_df, regioncolname)
    newcounts = newadata.to_memory().to_df()
    return newcounts, coords_df, labels_df['label']