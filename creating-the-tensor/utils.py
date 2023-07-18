import numpy as np
import matplotlib.pyplot as plt
import warnings

def fitSmoothingKernelBandwidth(full_traindict, total_trial_len):
    """Fits a spike smoothing kernel to spike train data using 
    the improved Sheather-Jones (ISJ) algorithm:
    
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
    “Kernel density estimation via diffusion.” 
    Annals of Statistics, Volume 38, Number 5, pp. 2916-2957, 2010.
    https://arxiv.org/pdf/1011.2602.pdf
    
    (see https://kdepy.readthedocs.io/en/latest/index.html 
    for more information on this implementation)
    
    
    ---------------
    Arguments:
    full_traindict: dict, {stimulus_direction: list of spike time arrays, one per trial}
    The bandwidth is computed for the stimulus direction that elicited the most spikes.
    total_trial_len: float or int, total length of a trial used for the trains; must
        use the same time unit as the spike times in `full_traindict`
    
    ---------------
    Returns:
    opt_bw: float, optimal bandwidth found
    fftkde: object, the fitted fftkde object, to be reused when evaluating the kernel
    """
    

    # 1) use stimulus direction with max n of spks to estimate optimal bandwidth
    maxn = -1
    for d, trains in full_traindict.items():

        data = np.concatenate(trains)
        assert max(data) < total_trial_len
        data.sort()
        n = data.size
        if n > maxn:
            maxn = n
            bestd = d

    data = np.concatenate(full_traindict[bestd])

    # 2) fit bw

    fftkde = FFTKDE(kernel='gaussian', bw='ISJ')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            fftkde = fftkde.fit(data)
        opt_bw = fftkde.bw

    except:
        #print(f'fftkde failed: {n} data points')
        opt_bw = None
        fftkde = None

    return opt_bw, fftkde #return fftkde object as well, to avoid refitting

def from0to1(arr):
    arr = np.asanyarray(arr)
    arr[np.isclose(arr,0)] = 1
    return arr

def subps(nrows,ncols,rowsz=3,colsz=4,d3=False,axlist=False):
    if d3:
        f = plt.figure(figsize=(ncols*colsz,nrows*rowsz))
        axes = [[f.add_subplot(nrows,ncols,ri*ncols+ci+1, projection='3d') for ci in range(ncols)] \
                for ri in range(nrows)]
        if nrows == 1:
            axes = axes[0]
            if ncols == 1:
                axes = axes[0]
    else:
        f,axes = plt.subplots(nrows,ncols,figsize=(ncols*colsz,nrows*rowsz))
    if axlist and ncols*nrows == 1:
        
        axes = [axes]
    return f,axes

def twx():
    ax = plt.subplot(111)
    return ax, ax.twinx()
    
def npprint(a,precision=3):
    with np.printoptions(precision=precision, suppress=True):
        print(a)
    return

def khatri_rao(matrices):
    """Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list of ndarray

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the
        product.

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))

import scipy as sp
import warnings
def computeResponseStats(traindict, ISI_Nspks, stats_ISI_len, trial_len, verbose=False):
    
    """Runs statistical tests to compare firing rates between the ISI and a given stimulus,
    for any period within the stimulus trial with the same length as the ISI.
    It performs two comparisons against the ISI FR: one using the maximum FR found for that 
    interval length across stimulus trials; and another using the minimum FR.
    
    The following one-sided tests are run:
    Mann-Whitney U
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    
    and Wilcoxon:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    
    Arguments:
    
    traindict: dict, {stimulus_direction: list of spike time arrays (ms), one per trial}
    ISI_Nspks: dict, {stimulus_direction: list of spike counts, one per trial}
    stats_ISI_len: float, length of the ISI interval, in secs, used to compute ISI_Nspks
    trial_len: float, total length of each trial, in secs
    
    Returns:
    stats_results: dict, for each of 'min-interval' and 'max-interval', contains a dict
     containing, for each stimulus direction, the p-values found for each tests, as well
     as the FRs for the stimulus and the ISI 
    """
    mydirs = traindict.keys()
    stats_results = {}



    interval_Nspks_for_stats = {'min':{}, 'max':{}}
    for d in mydirs:
    
        #combine all trains
        data = np.concatenate(traindict[d])
        counts,bins = np.histogram(data,np.arange(0,trial_len+stats_ISI_len,stats_ISI_len))

        maxfr = -1
        minfr = np.inf
        for i in range(0,counts.size):
            fr = counts[i]
            if fr > maxfr:
                maxi = i
                maxfr = fr
            if fr < minfr:
                mini = i
                minfr = fr


        for interval_type,i_,fr_ in [('min',mini,minfr), ('max',maxi,maxfr)]:

            interval_Nspks_for_stats[interval_type][d] = []

            for train in traindict[d]:
                spks_within_interval = train[(train >= stats_ISI_len*i_) & (train < stats_ISI_len*(i_+1))]
                interval_Nspks_for_stats[interval_type][d].append( spks_within_interval.size )

            interval_Nspks_for_stats[interval_type][d] = np.array(interval_Nspks_for_stats[interval_type][d])
#             print(d,interval_type,interval_Nspks_for_stats[interval_type][d])
            assert len(interval_Nspks_for_stats[interval_type][d]) == len(traindict[d])
            assert interval_Nspks_for_stats[interval_type][d].sum() == fr_


    for trial_Nspks_for_stats_,pval_type,alternative in [
                                           (interval_Nspks_for_stats['max'],'maxinterval-pval','greater'),
                                           (interval_Nspks_for_stats['min'],'mininterval-pval','less')]:
        stats_results[pval_type] = {}
        

        for d in mydirs:
            dir_grayspks = ISI_Nspks[d]
            dir_stimspks = trial_Nspks_for_stats_[d]
            assert len(dir_stimspks) == len(dir_grayspks)
            if len(dir_stimspks) * len(dir_grayspks) == 0:
                stats_results[pval_type][d] = (np.inf,0,0)
                stats_results[pval_type][d] = {'MANNWHITNEY':np.inf,'WILCOXON':np.inf,'isiFR':0,'stimFR':0}
                if verbose: print(f'{d} no spikes')
                continue

            try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                _, pval = sp.stats.mannwhitneyu(dir_stimspks, dir_grayspks, alternative=alternative)
            except:                 
                pval = np.inf

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",category=RuntimeWarning)
                    warnings.simplefilter("ignore",category=UserWarning)
                    _, wpval = sp.stats.wilcoxon(dir_stimspks, dir_grayspks, alternative=alternative)
            except:
                wpval = np.inf


            stats_results[pval_type][d] = {'MANNWHITNEY':pval,'WILCOXON':wpval,'isiFR':dir_grayspks.mean()/stats_ISI_len,'stimFR':dir_stimspks.mean()/stats_ISI_len}
            if verbose: print(f'{d} {pval_type} ({pval:.3f},{wpval:.3f}), gray={dir_grayspks.mean()/stats_ISI_len:.2f}, stim={dir_stimspks.mean()/stats_ISI_len:.2f}')
    
    return stats_results