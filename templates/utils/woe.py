# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.special import logit, expit

from sklearn.ensemble import RandomForestClassifier

plt.style.use('default')
plt.rcParams['axes.grid']=True
plt.rcParams['axes.axisbelow']=True
plt.rcParams['grid.color']='.8'

# COMMAND ----------

def inv_logit(x, logbase=None):
    if not logbase:
        return expit(x)
    else:
        return 1/(1+logbase**(-x))

# COMMAND ----------

def describe_data_g_targ(dat_df, target_var, logbase=np.e):
    num=dat_df.shape[0]
    n_targ = sum(dat_df[target_var]==1)
    n_ctrl = sum(dat_df[target_var]==0)
    assert n_ctrl+ n_targ == num
    base_rate = n_targ/num
    base_odds = n_targ/n_ctrl
    lbm = 1/np.log(logbase)
    base_log_odds = np.log(base_odds)*lbm
    nll_null = -(dat_df[target_var] * np.log(base_rate)*lbm + (1-dat_df[target_var])*np.log(1-base_rate)*lbm).sum()
    logloss_null = nll_null/num
    #min_bin_ct = np.ceil(1/base_rate).astype(int)
    
    print("Number of records:", num)
    print("Target count:", n_targ)
    print("Target rate:",base_rate)
    print("Target odds:",base_odds)
    print("Target log odds:",base_log_odds)
    print("Null model negative log-likelihood:",nll_null)
    print("Null model LogLoss:",logloss_null)
    
    print("")
    return {'num':num, 'n_targ':n_targ, 'base_rate':base_rate, 'base_odds':base_odds, 
            'base_log_odds':base_log_odds, 'nll_null':nll_null, 'logloss_null':logloss_null}

# COMMAND ----------

def get_numeric_cutpoints(notmiss_df, n_cuts=2, binner='qntl', min_bin_size=100):
    targ_var, p_var , *_ = notmiss_df.columns
    
    if (binner=='gini' or binner=='entropy') and (n_cuts>1):
        dtree = RandomForestClassifier(n_estimators=1, criterion=binner, max_leaf_nodes = n_cuts, bootstrap=False, min_samples_leaf=min_bin_size, max_features = None)
        dtree.fit(notmiss_df[[p_var]], notmiss_df[targ_var])
        tree0 = dtree.estimators_[0].tree_ 
        leafmask = (tree0.feature !=2)
        prog_cuts = pd.Series(tree0.threshold[leafmask], index=tree0.children_left[leafmask]).sort_index()
        var_cuts_w_min = pd.concat([pd.Series(notmiss_df[p_var].min()), prog_cuts])
    elif (binner=='unif'):
        bin_edges = np.linspace(notmiss_df[p_var].min(), notmiss[p_var].max(), n_cuts+1)
        var_cuts_w_min = pd.Series(bin_edges[:-1])
    elif (binner=='hist'):
        _, bin_edges = np.histogram(notmiss_df[p_var], bins=n_cuts)
        var_cuts_w_min = pd.Series(bin_edges[:-1])
    else:
        binner = 'qntl'
        var_cuts_w_min = notmiss_df[p_var].quantile(np.arange(0,1,1/np.where(n_cuts<1, 1, n_cuts)))
    
    return (var_cuts_w_min.round(12), binner)

# COMMAND ----------

def get_categories(notmiss_df, n_cats=2, binner=None, min_bin_size=100):
    targ_var, p_var, *_ = notmiss_df.columns
    
    var_cts = notmiss_df[p_var].value_counts()
    small_cats = var_cts[var_cts < min_bin_size].index.tolist()
    low_rank_cats = var_cts[(n_cats-1):].index.tolist()
    
    if (len(low_rank_cats) <= len(small_cats)) or (len(low_rank_cats)<=1):
        binner = 'none'
        other_set = set(low_rank_cats + small_cats)
        if len(other_set)<=1:
            incl_cats = var_cts.index.tolist()
        else:
            incl_cats = list(set(var_cts.index)-other_set)
    elif (binner == 'gini' or binner=='entropy') and (n_cats>1):
        dtree = RandomForestClassifier(n_estimators=1, criterion=binner, max_leaf_nodes = n_cats, bootstrap=False, min_samples_leaf=min_bin_size, max_features = None)
        dxx = pd.get_dummies(notmiss_df[p_var])
        dxx = dxx.loc[:, ~dxx.columns.isin(small_cats)]
        dtree.fit(dxx, notmiss_df[targ_var])
        feats = dtree.estimators_[0].tree_.feature
        incl_cats = dxx.columns[feats[feats!=2]].tolist()
    else:
        binner = 'rank'
        other_set = set(low_rank_cats + small_cats)
        incl_cats = list(set(var_cts.index) - other_set)
        
    return (incl_cats, binner)

# COMMAND ----------

def get_bin_edges(dat_df, n_cuts, binner='qntl', min_bin_size=100,  correct_cardinality=True):
    targ_var, p_var, *_ = dat_df.columns
    
    is_numeric = (not hasattr(dat_df[p_var], 'cat') and np.issubdtype(dat_df[p_var], np.number))
                  
    if is_numeric:
        srtd_var_cts = dat_df[p_var].value_counts(sort=False).sort_index()    
        desc_var_cts = srtd_var_cts.iloc[::-1]
                  
        cardinality0 = len(srtd_var_cts)
        if correct_cardinality:
            cum_desc_var_cts = desc_var_cts.cumsum().sort_index()
            cardinality = np.min([sum((srtd_var_cts.cumsum() >= min_bin_size) \
                                    & (cum_desc_var_cts>= min_bin_size)),(cum_desc_var_cts//min_bin_size).nunique()-1])       
        else:
            cardinality = cardinality0     
                  
    missing_ind = dat_df[p_var].isnull()
                  
    if (is_numeric and (cardinality>n_cuts)):
        notmiss_df = dat_df.loc[~missing_ind, [targ_var, p_var]]
        bin_edges, binner = get_numeric_cutpoints(notmiss_df, n_cuts, binner, min_bin_size)
    elif (is_numeric and (cardinality0>2)):
        binner = 'none'
        inc_bins = (srtd_var_cts >= min_bin_size)
                  
        cnt = 0
        for index, value in desc_var_cts.items():
            cnt += value
            if cnt >= min_bin_size:
                inc_bins[index] = True
                cnt = 0
                  
                  
        if ~inc_bins.iloc[0]:
            inc_bins.loc[inc_bins.idxmax()] = False
            inc_bins.iloc[0]=True
        bin_edges = srtd_var_cts[inc_bins].index.to_series().sort_values()                  
    else:
        dat_df[p_var] = dat_df[p_var].astype(str).str.strip()
        missing_ind = (missing_ind | (dat_df[p_var]==''))
        notmiss_df = dat_df.loc[~missing_ind, [targ_var, p_var]]
                  
        cat_sets, binner = get_categories(notmiss_df, n_cuts, binner, min_bin_size)        
        bin_edges = pd.Series(cat_sets)  
                  
    return (bin_edges, binner)

# COMMAND ----------

def gen_uwoesc_df(dat_df, bin_edges, binner=None, n_cuts=None, min_bin_size=100, laplace=1, laplaceY0='brc', compute_stats=False, neutralizeMissing=False, neutralizeSmBins=True, logbase=np.e):
    targ_var, p_var, *_ = dat_df.columns
    base_odds = sum(dat_df[targ_var]==1)/sum(dat_df[targ_var]==0)
                  
    if not binner:
        binner = 'custom'
        n_cuts = -1
     
    missing_ind = dat_df[p_var].isnull()
     
    if len(bin_edges)==0 or isinstance(list(bin_edges)[0], str) or isinstance(list(bin_edges)[0],set):
        var_type = 'C'
        dat_df[p_var] = dat_df[p_var].astype(str).str.strip()
        missing_ind = (missing_ind | (dat_df[p_var]==''))
    else:
        var_type = 'N'
                        
    notmiss_df = dat_df.loc[~missing_ind,:].copy()
    missing_df = dat_df.loc[missing_ind,:].copy()
        
    lbm = 1.0/np.log(logbase)
                        
    WOE_df = pd.DataFrame()
    if var_type=='C':
        missing_value = {''}
        input_cats = set()
        cat2set_dict={}
        for cat_set in bin_edges:
            if isinstance(cat_set, str):
                cat_set = {cat_set}
            input_cats = input_cats.union(cat_set)
            cat2set_dict.update(dict.fromkeys(cat_set, cat_set))
        other_cats = set(notmiss_df[p_var].unique()) - input_cats
         
        if len(other_cats)>0:
            cat2set_dict.update(dict.fromkeys(other_cats, {'Other'})) 
                     
        map_dict = {key:','.join(sorted(value)) for key, value in cat2set_dict.items()}
         
        notmiss_df['bin'] = notmiss_df[p_var].map(map_dict).astype(str)
        dfgb = notmiss_df.groupby('bin',as_index=True)
        bin_ct = dfgb[targ_var].count()
        if len(other_cats)<1 and missing_ind.sum()>0:
            bin_ct['other'] = 0
        WOE_df['bin_ct'] = bin_ct 
        WOE_df['bin_min'] = WOE_df.index.astype(str).to_series().str.split(',').apply(set)
        WOE_df['ranks'] = WOE_df.bin_ct.where(WOE_df.bin_min !='Other').rank(ascending=True)
        WOE_df.loc[WOE_df.bin_min == {'Other'}, 'ranks']=WOE_df.shape[0]
    else:
        missing_value = np.nan
        if isinstance(bin_edges, list):
            min_val = notmiss_df[p_var].min()
            if bin_edges[0] > min_val:
                print("Adding minimum value to cut string")
                bin_edges = [min_val] + bin_edges
            bin_edges = pd.Series(bin_edges)
                        
        srtd_bin_edges = bin_edges.sort_values()
        var_cuts_w_range = np.append(srtd_bin_edges.drop_duplicates().values, np.inf)
        notmiss_df['bin'] = pd.cut(notmiss_df[p_var], var_cuts_w_range, right=False)
        dfgb = notmiss_df.groupby("bin", as_index=True)
        WOE_df['bin_ct'] = dfgb[targ_var].count()
        WOE_df['bin_min'] = var_cuts_w_range[:-1]
        ranks_min = srtd_bin_edges.rank(method='min').drop_duplicates().values
        ranks_max = srtd_bin_edges.rank(method='max').drop_duplicates().values
        WOE_df['ranks'] = [(x1, x2) for x1, x2 in zip(ranks_min, ranks_max)]                
                    
    WOE_df['Y1'] = dfgb[targ_var].sum(min_count=0)
    WOE_df['Y1'] = WOE_df['Y1'].fillna(0).astype(int)
    WOE_df.sort_values(by=['ranks','Y1'], ascending=[True, False], inplace=True) 
    WOE_df.drop(WOE_df[WOE_df.bin_ct==0].index, inplace=True)
    missing_row = pd.DataFrame([[missing_df[targ_var].count(), missing_value,0, missing_df[targ_var].sum()]], columns=WOE_df.columns, index=pd.Index(['.'], name='bin'))     
    WOE_df = pd.concat([missing_row, WOE_df])
    WOE_df['Y0'] = WOE_df.bin_ct - WOE_df.Y1    
    WOE_df['bin_pct'] = WOE_df.bin_ct/WOE_df.bin_ct.sum()
    WOE_df['targ_rate'] = WOE_df.Y1/np.where(WOE_df.bin_ct==0, 1, WOE_df.bin_ct)
    
    if laplaceY0=='bal':
        laplaceY0=laplace
        base_odds = 1
    elif laplaceY0=='brc':
        laplaceY0 = laplace/base_odds
            
    if laplace<=0:
        pr_rate = 1.0/(1.0+np.exp(-np.log(base_odds)))
        null_event_fill_Y1 = pr_rate
        null_event_fill_Y0 = null_event_fill_Y1/base_odds
        Y1_ct = np.maximum(WOE_df.Y1, null_event_fill_Y1)
        Y0_ct = np.maximum(WOE_df.Y0, null_event_fill_Y0)
    else:
        Y1_ct = WOE_df.Y1+laplace
        Y0_ct = WOE_df.Y0+laplaceY0
        
    WOE_df['p_XgY1'] = Y1_ct/Y1_ct.sum()
    WOE_df['p_XgY0'] = Y0_ct/Y0_ct.sum()
    WOE_df['WOE'] = np.log(WOE_df.p_XgY1/WOE_df.p_XgY0).fillna(0)*lbm
    
    if neutralizeSmBins:
        WOE_df.loc[WOE_df.bin_ct<min_bin_size, 'WOE'] = 0
        
    if neutralizeMissing:
        WOE_df.loc[(WOE_df.bin_min.isna()| (WOE_df.bin_min== missing_value)), 'WOE'] = 0
        
    bins_out = WOE_df.shape[0]
    WOE_df['var_name'] = p_var
    WOE_df['var_type'] = var_type
    WOE_df['binner'] = binner 
    WOE_df['req_cuts'] = n_cuts
    WOE_df['bins'] = bins_out
    WOE_df['bin_idx'] = list(range(bins_out))
        
    WOE_df.index = WOE_df.index.astype(str)
    WOE_df.reset_index(inplace=True)
    WOE_df.set_index(['var_name','var_type','binner','req_cuts','bins','bin_idx'], inplace=True)
    
    if compute_stats:
        WOE_df['KLY1Y0'] = WOE_df.p_XgY1 * WOE_df.WOE
        WOE_df['KLY0Y1'] = WOE_df.p_XgY0 * -WOE_df.WOE
        
        WOE_df['IV'] = WOE_df['KLY1Y0']+WOE_df['KLY0Y1']
        WOE_df['nIV'] = WOE_df['IV']/np.log(bins_out)*lbm
        WOE_df['bin_pred'] = 1.0/(1.0+logbase**(-(np.log(base_odds)*lbm+WOE_df.WOE)))
        
    return(WOE_df)

# COMMAND ----------

def gen_woe_df(dat_df, p_var, targ_var, n_cuts=2, laplace=1, min_bin_size=100, binner='qntl', laplaceY0='brc', compute_stats=False,
               neutralizeMissing=False, neutralizeSmBins=True, correct_cardinality=True, logbase=np.e):
    
    """    
    Generate a weight of evidence dataframe
    ***
    Parameters
    -------------
    dat: DataFrame --- A datafram contains both the predictor and target variable
    p_var: String --- the predictor varaible
    targ_var: String: target variable, corresponding to a binary variable in dat
    n_cuts: int, optional (default=2) the number of data splots. Generally creates n_cuts+1 bins 
    laplace: int, float, optinal (default=0) --- Additive smoothing parameter used when estimating target (V=1) distribution probabilities
    min_bin_size: int, optional (default=1) --- Minimum bin size allowed. If created, bins with fewer than min_bin_size instances will have WOE=0 if neutralizeSmBin=True
    binner: String, optional (default='qnt1')
        the splitting algorithm used --- qntl: quantile binning is performed
                                     --- gini: gini impurity criterion is used for recursive partitioning in decision tree
                                     --- entropy: information gain criterion for split quality is used
    laplaceY0:int, float, string, optional ='brc'
         - if int ot float: the value is used for additive smoothing
         - if bal or balanced, laplace parameter is used
         - if 'brc' or base rate corrected, then laplace parameter devided by base_odd is used Makes additive smoothing respect the prior/base rate of the targetrt in the dataset.
    compute_stats: bool, optional(default=Fasel)
        Toggle for cimputing stats, KL divergens, IVs, bin prediction, etc
    neutralizeMissing: bool, optional (default=False)
        Toggle to set WOE value for missing value bins equal to zero
            
    ***    
    
    Returns
    ------
    WOE_df: DataFrame
    """
    dat_df = dat_df.loc[:, [targ_var, p_var]]
    bin_edges, binner = get_bin_edges(dat_df, n_cuts, binner, min_bin_size, correct_cardinality)
    
    WOE_df = gen_uwoesc_df(dat_df, bin_edges, binner, n_cuts, min_bin_size,laplace, laplaceY0, compute_stats, neutralizeMissing, neutralizeSmBins, logbase)
                        
    return (WOE_df)

# COMMAND ----------

def uwoesc_plot(WOE_df,targ_var, sort_values=False, var_scale='def', top_n=None, sep_bar=False, textsize=10, figsize=(8.2,4)):
    p_var = WOE_df.index[0][0]
    compute_stats='IV' in WOE_df.columns
    
    if top_n:
        WOE_df = WOE_df.iloc[:top_n+1,:]
    
    n_bins = WOE_df.index.get_level_values('bins')[0]
    
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig.tight_layout(pad=4.0)
    fig.text(0.5,0.92, p_var, ha='center', size=24)
    xticks = np.arange(0, len(WOE_df.WOE))
    ax1_xticks = xticks
    axs[1].axhline(0, color='.8', lw=2.5)
    ave_bin_width = 1
    woeCol = 'purple'
    woeAlpha = .6
              
    if isinstance(WOE_df.bin_min[1], set): 
        if sort_values:
            mask = (WOE_df.bin_min=='Other')
            WOE_df = pd.contact([WOE_df[~mask].sort_values('bin', na_position='First'), WOE_df[mask]])
        xlabel = 'bin_category'
        xticklabels = WOE_df.bin.tolist()
        ax1_xticklabels = xticklabels[1:]
        xtick_offset = 0
        text_offset = -.4
        barwidth = .8
        axs[1].bar(ax1_xticks[:-1], WOE_df.WOE[1:], width=barwidth, lw=1.5, fc=mpl.colors.to_rgb(woeCol)+(.3,), ec=mpl.colors.to_rgb(woeCol)+(woeAlpha,))
        axs[1].set_xticks(ax1_xticks[:-1])
    else:
        if WOE_df.bin_min[1:].apply(float.is_integer).all():
            xticklabels = WOE_df.bin_min.map('{:.0f}'.format).tolist()
        else:
            first_dec_digit = np.floor(np.log10(WOE_df.bin_min[1:].abs(), where=(WOE_df.bin_min[1:]!=0)))
            xtickdigits = np.min([3,np.nanmax([1,int(1-first_dec_digit.min())])])
            xticklabels = WOE_df.bin_min.apply(lambda x:'{:.{}f}'.format(x, xtickdigits)).tolist()
        xlabel = 'bin_cutpoints'
        ax1_xticklabels = xticklabels[1:] + ['max']
        xtick_offset =-.5
        text_offset = 0
        barwidth = 1
        if var_scale == 'orig' and len(WOE_df.bin_min)>2:
            ave_bin_width = np.nanmin([(WOE_df.bin_min[2:] - WOE_df.bin_min[2:].shift()).mean()
                                         ,(WOE_df.bin_min[-1] - WOE_df.bin_min[1])/(n_bins-1)])
            x_init = np.max([WOE_df.bin_min[1], WOE_df.bin_min[2] - 2*ave_bin_width])
            ax1_xticks = np.array([x_init] + WOE_df.WOE[2:].tolist()+[WOE_df.bin_min[-1] + 2*ave_bin_width])
            ax1_xticklabels[0] = 'min'
        axs[1].step(ax1_xticks, [WOE_df.WOE[1]] + WOE_df.WOE[1:].tolist(), color=woeCol, label="WOE", alpha=woeAlpha)
        axs[1].set_xticks(ax1_xticks)
              
    ra = .42
    if sep_bar:
        axs[0].bar(xticks+.12, WOE_df['p_XgY1'], width=.6, label=targ_var+'=1', facecolor=(1,0,0,ra))
        axs[0].bar(xticks-.08, WOE_df['p_XgY0'], width=.6, label=targ_var+'=0', fc=(1,0,0,ra/(ra+1)))
    else:
        axs[0].bar(xticks+.12, WOE_df['p_XgY1'], width=barwidth, label=targ_var+'=1', facecolor=(1,0,0,ra))
        axs[0].bar(xticks-.08, WOE_df['p_XgY0'], width=barwidth, label=targ_var+'=0', fc='b', alpha=ra/(ra+1))
    axs[0].set_xticks(xticks+xtick_offset)
    axs[0].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylim((0, np.where(axs[0].get_ylim()[1]<2/(n_bins-1), 2/(n_bins-1), axs[0].get_ylim()[1])))
    axs[0].set_ylabel("Probability Density")
    axs[0].legend(frameon=True, framealpha=1)

    axs[1].axhline(WOE_df.WOE[0], linestyle=':', color=woeCol, alpha=woeAlpha)
    axs[1].set_xticklabels(ax1_xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    axs[1].set_xlabel(xlabel)
    axs[1].set_xlim(left=axs[1].get_xlim()[0]-.7*ave_bin_width)
    axs[1].set_ylim((np.where(axs[1].get_ylim()[0]>-1, -1, axs[1].get_ylim()[0])
                  , np.where(axs[1].get_ylim()[0]<1, 1, axs[1].get_ylim()[1])))
    axs[1].set_ylabel("WOE (est. marginal log-odds)")

    if compute_stats:
        for i, x_pt in enumerate(ax1_xticks[:-1]):
            axs[1].text(x_pt+text_offset, WOE_df.WOE.values[i+1]+.01*np.diff(axs[1].get_ylim()), str(np.round((100*WOE_df.bin_pred.values)[i+1],2))+'%', color=woeCol, size=textsize)

        axs[1].text(axs[1].get_xlim()[0], WOE_df.WOE.values[0]+.01*np.diff(axs[1].get_ylim()), str(np.round((100*WOE_df.bin_pred.values)[0],2))+'%', color=woeCol, size=textsize)

        num = WOE_df.bin_ct.sum()
        n_targ = sum(WOE_df.Y1)
        MetricList = ["# of Records = {:.0f}".format(num), \
                      "# of Targets = {:.0f}".format(n_targ), \
                      "# Missing = {:.0f}".format(WOE_df.iloc[0].bin_ct) \
                    , "Base Rate = {:3f}%".format(100*(n_targ/num)) \
                    , "RelEntropyY1Y0 = {:5f}".format(sum(WOE_df.KLY1Y0)) \
                    , "RelEntropyY0Y1 = {:5f}".format(sum(WOE_df.KLY0Y1)) \
                    , "InfoVal = {:5f}".format(sum(WOE_df.IV)) \
                    , "nInfoVal = {:5f}".format(sum(WOE_df.nIV))]

        ylocs = np.arange(.97, 0, -.05)[:len(MetricList)]
        for yloc, sm in zip(ylocs, MetricList):
            axs[1].annotate(sm, xy=(1.02, yloc), xycoords='axes fraction')
              
    plt.show()
    return WOE_df

# COMMAND ----------

def univariate_sc_plot(dat, p_var, targ_var, n_cuts=2, laplace=1, min_bin_size=1, binner='qntl', \
                       compute_stats=True, sort_values=False, var_scale='def', top_n=None, sep_bar=False, \
                       textsize=10, figsize=(8.2,4), **kwargs):
    WOE_df = gen_woe_df(dat, p_var, targ_var, n_cuts, laplace, min_bin_size, binner, compute_stats=compute_stats, **kwargs)
       
    return uwoesc_plot(WOE_df, targ_var, sort_values=sort_values, var_scale=var_scale \
                     , top_n=top_n, sep_bar=sep_bar, textsize=textsize, figsize=figsize)
