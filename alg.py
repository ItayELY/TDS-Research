from IPython.display import FileLink, display, Image, IFrame, display_pdf, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def get_corrs_dict(df, pairs, g_attr):
    corrs = {}
    for x, y in pairs:
        corr_all = stats.pearsonr(df[x], df[y])
        corrs[x, y] = (corr_all.statistic, {})
    for g in set(df[g_attr]):
        
        df_g = df[df[g_attr] == g]
        # if g == 10:
        #     print(df_g)
        for x, y in pairs:
            if len(df_g[x]) == 1:
                continue
            corr_g = stats.pearsonr(df_g[x], df_g[y])
            corrs[x, y][1][g] = corr_g.statistic
    return corrs

def get_simpson(df, g_attr, attrs=None):
    if not attrs:
        attrs = [c for c in df.columns if c != g_attr and df[c].dtype == 'float64']
    # print(df[g_attr].dtype)
    # if df[g_attr].dtype in ['float64', 'int64']:
    # #   _, bins = pd.qcut(df[g_attr], 5, retbins=True)
    # #   print('hi')
    #   df[f'{g_attr}_binned'] = pd.qcut(df[g_attr], 5, duplicates='drop')
    #   g_attr = f'{g_attr}_binned'
    # #   print('hi')
    
    if not attrs:
        attrs = [c for c in df.columns if c != g_attr and df[c].dtype == 'float64']
    pairs = [(a, b) for idx, a in enumerate(attrs) for b in attrs[idx + 1:]]
    for x, y in pairs:
        
        df[x]=df[x].fillna(df[x].mean())
        df[y]=df[y].fillna(df[x].mean())
        # print(np.isnan(df[x]).any())
        # print(np.isnan(df[y]).any())

    corrs = get_corrs_dict(df, pairs, g_attr)
    # for (att1, att2), (corr, groups) in corrs.items():
    #     print(f'correlation of {att1}, {att2}: {corr}')
    #     for g, corr_g in groups.items():
    #         print(f'    and in group {g_attr}={g}: {corr_g}')
    results = {}
    for (att1, att2), (corr, groups) in corrs.items():
        # revs = []
        diffs = {}
        for g, corr_g in groups.items():
            # if corr_g*corr<0:
                # revs.append(g)
            diffs[g] = abs(corr - corr_g)
        score = sum(diffs.values())/len(diffs.values())
        results[att1, att2] = (score, diffs)
        # print(f'correlation of ({att1}, {att2}),\n\tdiffs: {diffs}\n\tscore: {score}')
        # if len(revs) > 0:
            # print(f'correlation of {att1}, {att2} is reversed in groups {revs}.')
    results = sorted(results.items(), key=lambda x:-x[1][0])
    res = results[0]
    contributions = {}
    for g in set(df[g_attr]):
        diffs = {}
        df_exc = df[df[g_attr] != g]
        
        # if g == 10:
        #     print(df_g)
        (x, y) = res[0]
        # print(x)
        corr_all = stats.pearsonr(df_exc[x], df_exc[y]).statistic
        for g_exc in set(df_exc[g_attr]):
            df_g = df_exc[df_exc[g_attr] == g_exc]
            corr_g = stats.pearsonr(df_g[x], df_g[y]).statistic
            diffs[g_exc] = abs(corr_all - corr_g)
        # print(diffs)
       
           
        score = sum(diffs.values())/len(diffs.values())
        contributions[g] = abs(res[1][0] - score)
    # print(contributions)

    # print(res)
    exp_g = max(contributions, key=contributions.get)
    df_not_exp_g = df[df[g_attr] != exp_g]
    df_exp_g = df[df[g_attr] == exp_g]
    fig, ax =plt.subplots(1,3, layout='constrained', figsize=(16,8))
    sns.regplot(x=df[res[0][0]], y=df[res[0][1]], ax=ax[1])
    sns.scatterplot(x=df[res[0][0]], y=df[res[0][1]], hue=df[g_attr], ax=ax[0])
    sns.regplot(x=df_not_exp_g[res[0][0]], y=df_not_exp_g[res[0][1]], ax=ax[2])
    # fig.suptitle(f'Found a Simpson\'s Paradox Occurance:\nreversed correlation between {res[0][0]} and {res[0][1]}\nwhen grouped by {g_attr}.\n\
                #  \nParticularly noticable in {g_attr}={exp_g}')
    fig.suptitle(f'Pattern bias detected in the correlation between \n{res[0][0]} and {res[0][1]}\n\n The bias is caused primarily by the outliered group {g_attr}={exp_g}')
    ax[0].set_title(f'\nall data')
    ax[1].set_title(f'\npattern of all data')
    if len(g_attr) > 15:
        p1 = g_attr[:14]
        p2 = g_attr[14:]
        g_attr = p1 + '\n' + p2
    ax[2].set_title(f'\npattern of \n{g_attr}!={exp_g}')
    plt.legend(fontsize='4', title_fontsize='12')
    plt.show()