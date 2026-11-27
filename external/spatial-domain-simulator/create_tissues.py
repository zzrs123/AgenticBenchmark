import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from typing import Optional, Union
from itertools import product


def adjust_coords(
        coords: np.ndarray,
        scaling: Optional[Union[int, list]] = 100,
        offset: Optional[Union[int, list]] = 0) -> np.ndarray:
    '''
    normalises coords to lie from 0 to 100, with optional scaling and offset.

    Args:
        coords (np.ndarray): array of shape (n_coords, 2) to be adjusted
        scaling (Optional[Union[int, list]]): optional scaling parameter(s)
        offset (Optional[Union[int, list]]): optional offset parameter(s)

    Returns:
        np.ndarray: adjusted coordinates
    '''
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    newcoords = (coords - mins) / (maxs - mins)
    newcoords = scaling * newcoords + offset
    return newcoords


def find_closest(
        background: np.ndarray,
        highlights: np.ndarray,
        labels: np.ndarray) -> np.ndarray:
    '''
    transfers labels from highlights to closest background points.

    Args:
        background (np.ndarray): background points
        highlights (np.ndarray): highlight points
        labels (np.ndarray): label for each highlight point

    Returns:
        np.ndarray: label for each background point
    '''
    distances = cdist(highlights, background)
    min_index = distances.argmin(axis=1)
    bglabels = np.ones((background.shape[0],), dtype=int) # make background spots ==1 
    bglabels[min_index] = labels + 2 # make labels start at 2 instead of 0
    return bglabels


def create_background(
        sd,
        n_bg,
        coords,
        labels,
        grid_layout) -> tuple:
    '''
    creates background for dataset (tissue-ify)
    '''
    if grid_layout:
        tempbg = np.array(list(product(range(int(np.sqrt(n_bg))), range(int(np.sqrt(n_bg))))))
    else:
        rng = np.random.RandomState(sd)
        tempbg = rng.rand(n_bg, 2)
    bg = adjust_coords(coords=tempbg)
    newlabels = find_closest(background=bg, highlights=coords, labels=labels)
    return bg, newlabels


def smooth_domains(coords, labels, n_neighs=50):
    '''
    refines coordinates by smoothing (majority vote)
    '''
    adj = pd.DataFrame(squareform(pdist(coords)))
    
    def refine(pred, adj_df, nbs):
        refined = []
        for ii, self_pred in enumerate(pred):
            adj_tmp = adj_df.loc[ii, :].sort_values()
            nbs_df = adj_tmp[0:nbs+1]
            nbs_pred = pred.loc[nbs_df.index]
            label_count = nbs_pred.value_counts()

            if self_pred not in label_count.index or label_count.loc[self_pred] <= nbs/2:
                refined.append(label_count.idxmax())
            else:
                refined.append(self_pred)
        return pd.DataFrame(refined)

    newlabels = refine(pred=pd.Series(labels), adj_df=adj, nbs=n_neighs)
    return coords, newlabels.to_numpy()


# inspiration from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def make_dataset(
        shape: str,
        n_samples: Optional[int] = 20000,
        n_bg: Optional[int] = 2000,
        scaling: Optional[Union[int, list]] = 90,
        offset: Optional[Union[int, list]] = 5,
        include_background: Optional[bool] = True,
        include_smoothing: Optional[bool] = True,
        grid_layout: Optional[bool] = False) -> tuple:
    '''
    creates scikit-learn inspired datasets.

    Args:
        shape (str): one of "stripes", "circles", "moons", "varied", "blobs", "2blobs" 
        n_samples (Optional[int]): number of samples to draw for shapes (density, if including bg)
        n_bg (Optional[int]): number of cells in dataset
        scaling (Optional[Union[int, list]]): optional scaling parameter(s) for shape position
        offset (Optional[Union[int, list]]): optional offset parameter(s) for shape position
        include_background (Optional[bool]): whether to include background points
        include_smoothing (Optional[bool]): whether to smooth labels transferred to background
        grid_layout (Optional[bool]): whether to simulate spots in a grid rather than uniformly random distributed (default: False)
    '''
    import sklearn.datasets
    seed1 = 30
    seed2 = 170

    if shape == 'stripes':
        if grid_layout:
            stripes_coords = np.array(list(product(range(int(np.sqrt(n_bg))), range(int(np.sqrt(n_bg))))))
        else:
            rng = np.random.RandomState(seed2)
            stripes_coords = rng.rand(n_bg, 2)
        stripes_coords = adjust_coords(coords=stripes_coords)
        stripes_labels = np.array([4 if (row[0]+int(row[1]/10)-5) > 80
                                   else 3 if (row[0]+int(row[1]/10)-5) > 45
                                   else 2 if (row[0]+int(row[1]/10)-5) > 20
                                   else 1 for row in stripes_coords])
        dataset = (stripes_coords, stripes_labels)

    elif shape == 'circles':
        dataset = sklearn.datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.15, random_state=seed1)
        dataset = adjust_coords(coords=dataset[0], scaling=scaling, offset=offset), dataset[1]

    elif shape == 'moons':
        dataset = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15, random_state=seed1)
        dataset = adjust_coords(coords=dataset[0], scaling=94, offset=3), dataset[1]

    elif shape == 'varied':
        dataset = sklearn.datasets.make_blobs(n_samples=n_samples, centers=[[0, 4], [-4, -4], [5, -1]], cluster_std=[1.5, 3, .75], random_state=seed2)
        dataset = adjust_coords(coords=dataset[0], scaling=98, offset=-3), dataset[1]

    elif shape == 'blobs':
        dataset = sklearn.datasets.make_blobs(n_samples=n_samples, centers=[[0, 4], [-3, -3] ,[4, -2]], cluster_std=2, random_state=seed2)
        dataset = adjust_coords(coords=dataset[0], scaling=99, offset=0), dataset[1]

    elif shape == '2blobs':
        dataset = sklearn.datasets.make_blobs(n_samples=n_samples, centers=[[-3, -3] ,[3, 3]], cluster_std=1.5, random_state=seed2)
        dataset = adjust_coords(coords=dataset[0], scaling=105, offset=0), dataset[1]

    else:
        raise ValueError(f'unknown shape "{shape}". Shape should be any of "stripes", "circles", "moons", "varied", "blobs", or "2blobs".')

    if include_background and shape != 'stripes':
        dataset = create_background(sd=seed1, n_bg=n_bg, coords=dataset[0], labels=dataset[1], grid_layout=grid_layout)
    if include_background and include_smoothing:
        dataset = smooth_domains(coords=dataset[0], labels=dataset[1])

    coords = pd.DataFrame(dataset[0], columns=['x', 'y'])
    labels = pd.DataFrame(dataset[1], columns=['label'])

    return coords, labels


def make_stripes(n_bg=2000, divisions=(80, 45, 30), grid_layout=False):
    seed = 170

    if grid_layout:
        stripes_coords = np.array(list(product(range(int(np.sqrt(n_bg))), range(int(np.sqrt(n_bg))))))
    else:
        rng = np.random.RandomState(seed)
        stripes_coords = rng.rand(n_bg, 2)
    stripes_coords = adjust_coords(coords=stripes_coords)
    stripes_labels = np.array([4 if (row[0]+int(row[1]/10)-5) > max(divisions)
                               else 3 if (row[0]+int(row[1]/10)-5) > int(np.median(divisions))
                               else 2 if (row[0]+int(row[1]/10)-5) > min(divisions)
                               else 1 for row in stripes_coords])
    dataset = (stripes_coords, stripes_labels)

    coords = pd.DataFrame(dataset[0], columns=['x', 'y'])
    labels = pd.DataFrame(dataset[1], columns=['label'])

    return coords, labels


def make_straight_stripes(n_bg=4000, divisions=(65, 35), grid_layout=False):
    seed = 170

    if grid_layout:
        stripes_coords = np.array(list(product(range(int(np.sqrt(n_bg))), range(int(np.sqrt(n_bg))))))
    else:
        rng = np.random.RandomState(seed)
        stripes_coords = rng.rand(n_bg, 2)
    stripes_coords = adjust_coords(coords=stripes_coords)
    stripes_labels = np.array([3 if row[0] > max(divisions)
                               else 2 if row[0] > min(divisions)
                               else 1 for row in stripes_coords])
    dataset = (stripes_coords, stripes_labels)

    coords = pd.DataFrame(dataset[0], columns=['x', 'y'])
    labels = pd.DataFrame(dataset[1], columns=['label'])

    return coords, labels