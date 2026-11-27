import pandas as pd
import anndata


def get_spottypes(show=False):
    celltype_meta = pd.read_csv('resources/CellType_Metadata.tsv', sep='\t')
    celltypes = celltype_meta[celltype_meta['num_cells_postQC'] > 0]
    if show:
        print(celltypes[['Annotation', 'cell_class', 'Max_TopStruct', 'num_cells_postQC']])
    celltypes = celltypes[['Annotation', 'num_cells_postQC']].rename(columns={'Annotation': 'spotname', 'num_cells_postQC': 'count'})
    spottypes_df = celltypes.set_index('spotname').sort_values('count', ascending=False)
    
    return spottypes_df


def assign_spots(
        adata: anndata.AnnData,
        labels_df: pd.DataFrame,
        labelcolname: str,
        randomstate: int = 29) -> tuple:
    """
    Assign spots from adata to the points in labels_df.
    Returns new adata.

    Args:
        adata (anndata.Anndata): Anndata object containing spots
        labels_df (pd.DataFrame): df containing column "region" with regions (from adata) to assign from
        labelcolname (str): name of column in adata.obs containing region names
    """
    n_of_each_type = labels_df.value_counts('region')
    spots_df = []
    for rg, n_spots in n_of_each_type.items():
        cells_of_ct = adata.obs[adata.obs[labelcolname] == rg].copy()
        spots_df.append(cells_of_ct.sample(n_spots, random_state=randomstate))
    spots_df = pd.concat(spots_df)

    # ensure matching of cell names to labels
    order_dict = {}
    for rg in n_of_each_type.index:
        spots_order = spots_df[spots_df[labelcolname] == rg].index
        cts_order = labels_df[labels_df['region'] == rg].index
        order_dict = order_dict | dict(zip(spots_order, cts_order))

    spots_df = spots_df.reset_index().rename(columns={'index': 'selected_cell'})
    spots_df['spots_order'] = spots_df['selected_cell'].map(order_dict)
    spots_df = spots_df.sort_values('spots_order')
    spots_df = spots_df.set_index('spots_order')

    return adata[spots_df['selected_cell']]


def map_spottypes(labels_df, spottypes_df, seed):
    labels_df['region'] = labels_df['label'].astype(str).map(spottypes_df.loc[f'spottype_seed{seed}'].to_dict())
    return labels_df