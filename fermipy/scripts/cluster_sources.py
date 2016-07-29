# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import sys
import argparse
import yaml
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.csr import csr_matrix
from astropy.extern import six
from astropy.table import Table, Column


def read_table(filepath):
    """ Trivially read and return a table
    """
    tab = Table.read(filepath)
    return tab


def make_lat_lons(cvects):
    """ Convert from directional cosines to latitidue and longitude

    Parameters
    ----------
    cvects : directional cosine (i.e., x,y,z component) values

    returns (np.ndarray(2,nsrc)) with the directional cosine (i.e., x,y,z component) values
    """
    lats = np.degrees(np.arcsin(cvects[2]))
    lons = np.degrees(np.arctan2(cvects[0], cvects[1]))
    return np.hstack([lats, lons])


def make_cos_vects(ra_vect, dec_vect):
    """ Convert from RA and DEC values to directional cosines

    Parameters
    ----------
    ra_vect,dec_vect : np.ndarray(nsrc)  
       RA and DEC values

    returns (np.ndarray(3,nsrc)) with the directional cosine (i.e., x,y,z component) values
    """
    ra_rad = np.radians(ra_vect)
    dec_rad = np.radians(dec_vect)
    svals = np.sin(dec_rad)
    xvals = svals * np.sin(ra_rad)
    yvals = svals * np.cos(ra_rad)
    zvals = np.cos(dec_rad)
    cvects = np.vstack([xvals, yvals, zvals])
    return cvects


def find_matches_by_distance(cos_vects, cut_dist):
    """ Find all the pairs of sources within a given distance of each other

    Parameters
    ----------
    cos_vects : np.ndarray(e,nsrc)  
       Directional cosines (i.e., x,y,z component) values of all the sources

    cut_dist : float

    returns dict((int,int):float).  
       Each entry gives a pair of source indices, and the corresponding distance
    """
    dist_rad = np.radians(cut_dist)
    cos_t_cut = np.cos(dist_rad)
    nsrc = cos_vects.shape[1]
    matchDict = {}
    for i, v1 in enumerate(cos_vects.T):
        cos_t_vect = (v1 * cos_vects.T).sum(1)
        mask = cos_t_vect > cos_t_cut
        acos_t_vect = np.ndarray(nsrc)
        acos_t_vect[mask] = np.degrees(np.arccos(cos_t_vect[mask]))
        for j in xrange(0, i):
            if mask[j]:
                matchDict[(j, i)] = acos_t_vect[j]
            pass
        pass
    return matchDict


def find_matches_by_sigma(cos_vects, unc_vect, cut_sigma):
    """ Find all the pairs of sources within a given distance of each other

    Parameters
    ----------
    cos_vects : np.ndarray(3,nsrc)  
       Directional cosines (i.e., x,y,z component) values of all the sources

    unc_vect : np.ndarray(nsrc)  
       Uncertainties on the source positions

    cut_sigma : float

    returns dict((int,int):float).  
       Each entry gives a pair of source indices, and the corresponding sigma
    """
    dist_rad = np.radians(dist)
    cos_t_cut = np.cos(dist_rad)
    nsrc = cos_vects.shape[1]
    match_dict = {}
    acos_t_vect = np.ndarray(nsrc)
    sig_2_vect = unc_vect * unc_vect

    for i, v1 in enumerate(cos_vects.T):
        cos_t_vect = (v1 * cos_vects.T).sum(1)
        sig_2_i = sig_2_vect[i]
        acos_t_vect = np.degrees(np.arccos(cos_t_vect[mask]))
        total_unc = np.sqrt(sig_2_i + sig_2_vect)
        sigma_vect = acos_t_vect / total_unc
        mask = sigma_vect < cut_sigma
        for j in xrange(0, i):
            if mask[j]:
                match_dict[(j, i)] = sigma_vect[j]
            pass
        pass
    return match_dict


def fill_edge_matrix(nsrcs, match_dict):
    """ Create and fill a matrix with the graph 'edges' between sources.

    Parameters
    ----------
    nsrcs  : int 
       number of sources (used to allocate the size of the matrix)

    match_dict :  dict((int,int):float)
       Each entry gives a pair of source indices, and the corresponding measure (either distance or sigma) 

    returns numpy.ndarray((nsrcs,nsrcs)) filled with zeros except for the matches, which are filled with the
       edge measures (either distances or sigmas)
    """
    e_matrix = np.zeros((nsrcs, nsrcs))
    for k, v in match_dict.items():
        e_matrix[k[0], k[1]] = v
        pass
    return e_matrix


def make_rev_dict_unique(cdict):
    """ Make a reverse dictionary

    in_dict : dict(int:dict(int:True))
       A dictionary of clusters.   Each cluster is a source index and the dictionary of other sources in the cluster.    

    returns dict(int:dict(int:True))
       A dictionary pointing from source index to the clusters it is included in.

    """
    rev_dict = {}
    for k, v in cdict.items():
        if rev_dict.has_key(k):
            rev_dict[k][k] = True
        else:
            rev_dict[k] = {k: True}
        for vv in v.keys():
            if rev_dict.has_key(vv):
                rev_dict[vv][k] = True
            else:
                rev_dict[vv] = {k: True}
    return rev_dict


def make_clusters(span_tree, cut_value):
    """ Find clusters from the spanning tree

    Parameters
    ----------
    span_tree : a sparse nsrcs x nsrcs array
       Filled with zeros except for the active edges, which are filled with the
       edge measures (either distances or sigmas

    cut_value : float
       Value used to cluster group.  All links with measures above this calue will be cut.

    returns dict(int:[int,...])  
       A dictionary of clusters.   Each cluster is a source index and the list of other sources in the cluster.    
    """
    iv0, iv1 = span_tree.nonzero()

    # This is the dictionary of all the pairings for each source
    match_dict = {}

    for i0, i1 in zip(iv0, iv1):
        d = span_tree[i0, i1]
        # Cut on the link distance
        if d > cut_value:
            continue

        imin = int(min(i0, i1))
        imax = int(max(i0, i1))
        if match_dict.has_key(imin):
            match_dict[imin][imax] = True
        else:
            match_dict[imin] = {imax: True}

    working = True
    while working:

        working = False
        rev_dict = make_rev_dict_unique(match_dict)

        for k, v in rev_dict.items():
            # Multiple mappings
            if len(v) > 1:
                working = True
                cluster_idx = v.keys()[0]
                for vv in v.keys()[1:]:
                    try:
                        to_merge = match_dict.pop(vv)
                    except:
                        continue
                    try:
                        match_dict[cluster_idx].update(to_merge)
                        match_dict[cluster_idx][vv] = True
                    except:
                        continue
                    # remove self references
                    try:
                        match_dict[cluster_idx].pop(cluster_idx)
                    except:
                        pass

    # Convert to a int:list dictionary
    cdict = {}
    for k, v in match_dict.items():
        cdict[k] = v.keys()

    # make the reverse dictionary
    rdict = make_reverse_dict(cdict)
    return cdict, rdict


def select_from_cluster(idx_key, idx_list, measure_vect):
    """ Select a single source from a cluster and make it the new cluster key

    Parameters
    ----------
    idx_key : int
      index of the current key for a cluster

    idx_list : [int,...]
      list of the other source indices in the cluster

    measure_vect : np.narray((nsrc),float)
      vector of the measure used to select the best source in the cluster

    returns best_idx:out_list
      where best_idx is the index of the best source in the cluster and
            out_list is the list of all the other indices
    """
    best_idx = idx_key
    best_measure = measure_vect[idx_key]
    out_list = [idx_key] + idx_list
    for idx, measure in zip(idx_list, measure_vect[idx_list]):
        if measure < best_measure:
            best_idx = idx
            best_measure = measure
    out_list.remove(best_idx)
    return best_idx, out_list


def find_centroid(cvects, idx_list, weights=None):
    """ Find the centroid for a set of vectors

    Parameters
    ----------
    cvects : ~numpy.ndarray(3,nsrc) with directional cosine (i.e., x,y,z component) values

    idx_list : [int,...]
      list of the source indices in the cluster

    weights : ~numpy.ndarray(nsrc) with the weights to use.  None for equal weighting

    returns (np.ndarray(3)) with the directional cosine (i.e., x,y,z component) values of the centroid
    """
    if weights is None:
        weighted = cvects.T[idx_list].sum(0)
        sum_weights = float(len(idx_list))
    else:
        weighted = ((cvects * weights).T[idx_list]).sum(0)
        sum_weights = weights.sum(0)
    weighted /= sum_weights
    # make sure it is normalized
    norm = np.sqrt((weighted * weighted).sum())
    weighted /= norm
    return weighted


def find_dist_to_centroid(cvects, idx_list, weights=None):
    """ Find the centroid for a set of vectors

    Parameters
    ----------
    cvects : ~numpy.ndarray(3,nsrc) with directional cosine (i.e., x,y,z component) values

    idx_list : [int,...]
      list of the source indices in the cluster

    weights : ~numpy.ndarray(nsrc) with the weights to use.  None for equal weighting

    returns (np.ndarray(nsrc)) distances to the centroid (in degrees)
    """
    centroid = find_centroid(cvects, idx_list, weights)
    dist_vals = np.degrees(np.arccos((centroid * cvects.T[idx_list]).sum(1)))
    return dist_vals, centroid


def find_dist_to_centroids(cluster_dict, cvects, weights=None):
    """ Find the centroids and the distances to the centroid for all sources in a set of clusters

    Parameters
    ----------
    cluster_dict : dict(int:[int,...])  
         Each cluster is a source index and the list of other sources in the cluster.  

    cvects : np.ndarray(3,nsrc)  
       Directional cosines (i.e., x,y,z component) values of all the sources

    weights : ~numpy.ndarray(nsrc) with the weights to use.  None for equal weighting

    Returns
    ----------
    distances : ~numpy.ndarray(nsrc) with the distances to the centroid of the cluster.  0 for unclustered sources

    cent_dict : dict(int:numpy.ndarray(2)), dictionary for the centroid locations
    """
    distances = np.zeros((cvects.shape[1]))
    cent_dict = {}
    for k, v in cluster_dict.items():
        l = [k] + v
        distances[l], centroid = find_dist_to_centroid(cvects, l, weights)
        cent_dict[k] = make_lat_lons(centroid)
    return distances, cent_dict


def select_from_clusters(cluster_dict, measure_vect):
    """ Select a single source from each cluster and make it the new cluster key

    cluster_dict : dict(int:[int,])        
       A dictionary of clusters.   Each cluster is a source index and the list of other source in the cluster.    

    measure_vect : np.narray((nsrc),float)
      vector of the measure used to select the best source in the cluster

    returns dict(int:[int,...])  
       New dictionary of clusters keyed by the best source in each cluster
    """
    out_dict = {}
    for idx_key, idx_list in cluster_dict.items():
        out_idx, out_list = select_from_cluster(
            idx_key, idx_list, measure_vect)
        out_dict[out_idx] = out_list
    return out_dict


def make_reverse_dict(in_dict):
    """ Build a reverse dictionary from a cluster dictionary

    in_dict : dict(int:[int,])        
       A dictionary of clusters.   Each cluster is a source index and the list of other source in the cluster.    

    returns dict(int:int)
       A single valued dictionary pointing from source index to cluster key for each source in a cluster.
       Note that the key does not point to itself.
    """
    out_dict = {}
    for k, v in in_dict.items():
        for vv in v:
            if vv in out_dict:
                print("Dictionary collision %i" % vv)
            out_dict[vv] = k
    return out_dict


def filter_and_copy_table(tab, to_remove):
    """ Filter and copy a FITS table.

    tab :  FITS Table object 

    to_remove : [int ...}
        list of indices to remove from the table

    returns  FITS Table object
    """
    nsrcs = len(tab)
    mask = np.zeros((nsrcs), '?')
    mask[to_remove] = True
    inv_mask = np.invert(mask)
    out_tab = tab[inv_mask]
    return out_tab


def make_match_hist(match_dict, match_cut, nbins=50):
    """
    """
    hist = np.histogram(match_dict.values(), nbins, (0., match_cut))
    return hist


def make_rename_dict(rev_dict, src_names):
    """
    """
    ret_dict = {}
    for k, v in rev_dict.items():
        ok = src_names[k]
        ov = src_names[v]
        if ok != ov:
            ret_dict[str(ok)] = str(ov)
    return ret_dict


def main():
    # Argument defintion
    usage = "usage: %(prog)s [input]"
    description = "Collect all the new source"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('--input', type=argparse.FileType('r'),
                        help='Input fits file.')
    parser.add_argument('--output', type=argparse.FileType('w'),
                        help='Output file.')
    parser.add_argument('--dist', type=float, default=None,
                        help="Maximum clustering distance (degrees)")
    parser.add_argument('--sigma', type=float, default=None,
                        help="Maximum clustering seperation (sigma)")
    parser.add_argument('--full', action='store_true', default=False,
                        help='Use full set off matches for clustering.')
    parser.add_argument('--clobber', action='store_true',
                        default=False, help='Overwrite output files.')

    # Argument parsing
    args = parser.parse_args()

    if args.dist and args.sigma:
        print("Specify only one of --dist and --sigma")
        sys.exit()

    if args.dist:
        use_dist = True
        match_cut = args.dist
    elif args.sigma:
        use_dist = False
        match_cut = args.sigma
    else:
        print("Specify either --dist or --sigma")
        sys.exit()

    use_full = args.full

    # read table and get relevant columns
    tab = Table.read(args.input)
    ra_vect = tab['GLON'].data
    dec_vect = tab['GLAT'].data
    offset_vect = tab['offset'].data
    src_names = tab['Source_Name'].data
    #TS_vect = tab['ts'].data

    # Convert everything to directional cosines
    cvects = make_cos_vects(glon_vect, glat_vect)

    # Find matches
    if use_dist:
        if use_full:
            dict_match_cut = match_cut
        else:
            dict_match_cut = 180.
        matchDict = find_matches_by_distance(cvects, dict_match_cut)
    else:
        sigma_vect = tab['loc_err'].data
        matchDict = find_matches_by_sigma(cvects, match_cut)

    # Make a histogram of the match measure
    matchHist = make_match_hist(matchDict, match_cut)

    # Build a matrix of the edes and apply the MST algorithm
    e_matrix = fill_edge_matrix(len(glon_vect), matchDict)
    full_tree = csr_matrix(e_matrix)

    # Apply the spanning tree
    span_tree = csgraph.minimum_spanning_tree(full_tree)

    # Turn the MST into a dictionary of clusters
    if use_full:
        cDict, rDict = make_clusters(full_tree, match_cut)
    else:
        cDict, rDict = make_clusters(span_tree, match_cut)

    # Find the centroids of the cluster
    dist_to_cent, centroids = find_dist_to_centroids(
        cDict, cvects, tab['npred'].data)

    # Select the best source from each cluster
    sel_dict = select_from_clusters(cDict, dist_to_cent)

    # This is a map you can use to replace sources
    # It maps duplicate -> original
    rev_dict = make_reverse_dict(sel_dict)
    rename_dict = make_rename_dict(rev_dict, src_names)

    # Copy the table, filtering out the duplicates
    to_remove = rev_dict.keys()
    out_tab = filter_and_copy_table(tab, to_remove)

    # Write the output
    if args.output:
        out_tab.write(args.output, format='fits')
        out_idx = args.output.name.replace(".fits", "_idx_dict.yaml")
        out_rename = args.output.name.replace(".fits", "_name_dict.yaml")
        out_hist = args.output.name.replace(".fits", "_hist.yaml")
        if args.clobber:
            fout_idx = open(out_idx, 'w!')
            fout_rename = open(out_rename, 'w!')
            fout_hist = open(out_hist, 'w!')
        else:
            fout_idx = open(out_idx, 'w')
            fout_rename = open(out_rename, 'w')
            fout_hist = open(out_hist, 'w')

        fout_idx.write(yaml.dump(sel_dict))
        fout_idx.close()
        fout_rename.write(yaml.dump(rename_dict))
        fout_rename.close()

if __name__ == "__main__":
    main()
