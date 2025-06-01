import logging
import os
import argparse
import pickle

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set_context("notebook", font_scale=2)

BONFERRONI_CORRECTION = False  # for comparators
COMPARATOR_NAMES = {
    "Cond_Cov": "Covariate",
    "Cond_Outcome": "Outcome",
    "KCICovariateTest": "KCI $\dagger$",
    "ParametricChangeExplanation": "ParamY $\ddag$",
    "ParametricAccExplanation": "ParamLoss $\ddag$",
    "KCIOutcomeTestAgg": "KCI",
    "KCIOutcomeTest": "KCI $\dagger$",
    "TEVIMTest": "TE-VIM $\dagger$",
    "KSTest": "KS $\ddag$",
    # "DomainClassifierExplanation",
    "ScoreMethod": "Score $\ddag$",
    "KCICovariateTestAgg": "KCI",
    # "LinearMediationTest",
}

cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0,'Red'),(1,'Grey')])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize tests"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="replicates",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov', 'Agg'],
        help="type of detailed decomposition, either conditional outcome or covariate",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        help="significance level i.e. alpha for confidence intervals e.g. 0.05",
    )
    parser.add_argument(
        "--grouping-data",
        type=str,
        default="_output/grouping_data.csv",
        help="grouping data",
    )
    parser.add_argument(
        "--target-data-file",
        type=str,
        help="target data in csv file",
    )
    parser.add_argument(
        "--comparator-file",
        type=str,
        help="output of importance of comparators",
    )
    parser.add_argument(
        "--decomp-file",
        type=str,
        help="output of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="aggregate results from replicates if simulation",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/eval_model_perf.txt",
        help="results file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.comparator_file = args.comparator_file.replace("JOB", 
            str(args.job_idx))
    args.decomp_file = args.decomp_file.replace("JOB", 
            str(args.job_idx))
    args.target_data_file = args.target_data_file.replace("JOB", 
        str(args.job_idx))
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    return args

def parse_proposed_vars(var_str):
    mask = np.array(var_str.replace('(','').replace(')','').replace(' ','').split(',')) == 'False'
    mask_false_idx = np.where(mask)[0]
    mask_false = ["X%d" % (i+1) for i in mask_false_idx]
    return ",".join(mask_false)

def parse_comp_vars(var_str, num_p):
    all_vars = ["X%d" % (i+1) for i in np.arange(num_p)]  # 1-indexed
    excluded_indices = var_str.split(",")
    included_indices = [i for i in all_vars if i not in excluded_indices]
    return ",".join(included_indices)

def plot_aggregate(result_data, significance_level):
    """Show 1 - pvalue for aggregate tests
    Color of cell indicates whether covariate or outcome test is flagged
    Decision to flag is taken for the outcome shift if McEE test is rejected, pvalue_mcee <= significance_level
    Covariate shift additionally requires that independence tests are rejected
    """
    df = pd.read_csv(result_data)
    df_agg = df[(df.decomp == 'agg') & (df.est == 'onestep')]

    df_agg.loc[:, 'decision'] = df_agg.loc[:, 'pvalue_mcee'] <= significance_level  # decision to flag, 1 is flag and 0 is not flag
    # cov_idx = df_agg['vars']=='X'
    # if not df_agg.loc[cov_idx, 'pvalue_independence'].isna().all():
        # df_agg.loc[cov_idx, 'decision'] = (df_agg.loc[cov_idx, 'pvalue_independence'] <= significance_level) & (df_agg.loc[cov_idx, 'pvalue_mcee'] <= significance_level)

    df_agg.loc[:,'vars'] = df_agg.loc[:, 'vars'].replace('X','Covariate $\ddag$')
    df_agg.loc[:,'vars'] = df_agg.loc[:, 'vars'].replace('Y','Outcome $\ddag$')
    
    df_agg_pivot = df_agg.pivot(index=['decomp'], columns='vars', values='decision')
    df_agg_annot_pivot = df_agg.pivot(index=['decomp'], columns='vars', values='pvalue_mcee')
    df_agg_pivot = df_agg_pivot.reset_index(drop=True)
    df_agg_annot_pivot = df_agg_annot_pivot.reset_index(drop=True)
    df_agg_pivot.columns.name = ''
    print(df_agg_pivot.columns)
    print(df_agg_annot_pivot.columns)
    
    plt.figure(figsize=(5,1))
    ax = sns.heatmap(1 - df_agg_pivot, 
                    annot=1 - df_agg_annot_pivot, fmt='.2f', 
                    cmap=cmap, vmin=0, vmax=1,
                    yticklabels=False, linewidths=2,
                    #  cbar_kws={'label': 'pvalue'},
                    cbar = False,
        )
    ax.xaxis.tick_top()
    ax.set_title('Aggregate', pad=10)
    plt.savefig(result_data.replace('.csv','target_inference_agg.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_agg_annot_pivot, df_agg_pivot

def get_group_names(grouping_data):
    names = pd.read_csv(grouping_data)
    names.loc[:, 'var'] = 'X' + (names.loc[:, 'var_idx']+1).astype(str)
    names['group_vars'] = names.groupby('group_name')['var'].transform(lambda x: ",".join(x))
    names = names[['group_vars', 'group_name']].drop_duplicates()
    return names

def plot_detailed(result_data, decomp, names, significance_level):
    """Show pvalue for detailed tests for feature subsets
    Color of cell indicates whether the subset is flagged
    We decide to flag a subset for outcome shift if McEE test is not significant, pvalue > significance level
    Covariate shift additionally requires that independence test is rejected to be flagged
    """
    df = pd.read_csv(result_data)
    df_detail = df[(df.decomp == decomp) & (df.est == 'onestep')]
    df_detail.loc[:, 'vars'] = df_detail.loc[:, 'vars'].apply(parse_proposed_vars)
    df_detail.loc[:, 'decision'] = df_detail.loc[:, 'pvalue_mcee'] > significance_level  # decision to flag, 1 is flag and 0 is not flag
    # if decomp == 'Cond_Cov' and not df_detail['pvalue_independence'].isna().all():
    #     df_detail.loc[:, 'decision'] = (df_detail.loc[:, 'pvalue_independence'] <= significance_level) & (df_detail.loc[:, 'pvalue_mcee'] > significance_level)
    df_detail_annot = df_detail[['vars','pvalue_mcee']]
    df_detail = df_detail[['vars','decision']]
    def pivot_trim_df(df_detail):
        if names is not None:
            df_detail = df_detail.merge(names, left_on="vars", right_on="group_vars")
            df_detail = df_detail.drop(['vars','group_vars'], axis=1)
            df_detail.set_index('group_name', inplace=True)
            df_detail = df_detail.sort_values(by='group_name')
        else:
            df_detail.set_index('vars', inplace=True)
            df_detail = df_detail.sort_values(by='vars')
        df_detail.index.name = None
        return df_detail
    df_detail_annot = pivot_trim_df(df_detail_annot)
    df_detail = pivot_trim_df(df_detail)
    # df_detail.columns = ['SHIFT']
    print(df_detail.index)
    print(df_detail_annot.index)

    plt.figure(figsize=(2,3))
    ax = sns.heatmap(1 - df_detail, 
                    annot=df_detail_annot, fmt='.2f', 
                    cmap=cmap, vmin=0, vmax=1,
                    xticklabels=False, 
                    yticklabels=False,
                    cbar=False
                    )
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('Detailed $\dagger$')
    plt.savefig(result_data.replace('.csv', '_detail.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_detail_annot, df_detail

def plot_comparators_real(comp_data, target_data, decomp, names, significance_level):
    num_p = pd.read_csv(target_data).shape[1] - 1  # number of features
    
    df = pd.read_csv(comp_data)

    # Remove rows for CausalForestExplanation
    df = df[df['est'] != 'CausalForestExplanation']

    # Remove rows for DomainClassifierExplanation
    df = df[df['est'] != 'DomainClassifierExplanation']

    # Do Bonferroni correction on each est group
    df['significance_level'] = significance_level
    if BONFERRONI_CORRECTION:
        df['significance_level'] = df.groupby(['est'])['significance_level'].transform(lambda x: x / len(x))

    # Get excluded vars for TEVIM and KCI tests, as the vars column logs complement of the variable of interest
    explanation_tests = df.est.str.contains('KCIOutcomeTest|TEVIMTest|KCICovariateTest')
    df.loc[explanation_tests, 'vars'] = df.loc[explanation_tests, 'vars'].transform(lambda x: parse_comp_vars(x, num_p))

    # Decide test by comparing to significance level
    df.loc[:, 'significant'] = False
    df.loc[explanation_tests, 'significant'] = np.where(df.loc[explanation_tests, 'pvalue'] < df.loc[explanation_tests, 'significance_level'], False, True)
    df.loc[~explanation_tests, 'significant'] = np.where(df.loc[~explanation_tests, 'pvalue'] < df.loc[~explanation_tests, 'significance_level'], True, False)
    df['significant'] = (df.loc[:, 'significant']).astype(float)

    # Keep explanation tests to plot
    df = df.loc[explanation_tests]

    df = df.merge(names, left_on='vars', right_on='group_vars', how='left')

    # Keep individual var names if no group found
    df.loc[df['group_vars'].isna(), 'group_name'] = df.loc[df['group_vars'].isna(), 'vars']

    # Pivot df on vars and est columns
    df_pivot = df.pivot(index=['group_name'], columns='est', values='significant')
    df_annot_pivot = df.pivot(index=['group_name'], columns='est', values='pvalue')

    # Sort by vars
    df_pivot = df_pivot.reset_index()
    df_annot_pivot = df_annot_pivot.reset_index()

    df_pivot.set_index('group_name', inplace=True)
    df_annot_pivot.set_index('group_name', inplace=True)

    # Write to csv
    df_annot_pivot.to_csv(comp_data.replace('.csv', '_pivot.csv'), index=False)
    print(df_pivot.columns)
    print(df_annot_pivot.columns)

    explanation_tests_cols = df_pivot.columns.str.contains('KCIOutcomeTest|TEVIMTest|KCICovariateTest')
    df_pivot.loc[:, explanation_tests_cols] = 1 - df_pivot.loc[:, explanation_tests_cols]
    df_pivot.columns.name = None
    df_pivot.index.name = None
    if df_pivot.shape[1] == 1:
        plt.figure(figsize=(2,3))
    else:
        plt.figure(figsize=(4,3))
    df_pivot = df_pivot.rename(columns=COMPARATOR_NAMES)
    ax = sns.heatmap(df_pivot, 
                    annot=df_annot_pivot, fmt='.2f',
                    cmap=cmap, vmin=0, vmax=1,
                    yticklabels=decomp=='Cond_Cov',
                    cbar=False,
                    )
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('Baselines %s' % COMPARATOR_NAMES[decomp])
    plt.savefig(comp_data.replace('.csv', '_detail.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_annot_pivot, df_pivot

def concat_files_to_df(files, num_jobs):
    all_res = []
    for file in files:
        if num_jobs is not None:
            for job_idx in range(num_jobs):
                file_jobidx = file.replace("JOB", str(job_idx+1))
                if os.path.exists(file_jobidx):
                    all_res.append(pd.read_csv(file_jobidx, delimiter=',', quotechar='"'))
                else:
                    print("FILE MISSING", file_jobidx)
        else:
            all_res.append(pd.read_csv(file, delimiter=',', quotechar='"'))
    return pd.concat(all_res)

def plot_aggregate_simulation(df, result_file):
    df_agg = df[(df.decomp == 'agg') & (df.est == 'onestep')]
    df_agg.loc[:,'vars'] = df_agg.loc[:, 'vars'].replace('X','Covariate $\ddag$')
    df_agg.loc[:,'vars'] = df_agg.loc[:, 'vars'].replace('Y','Outcome $\ddag$')
    df_agg = df_agg.groupby(by=['level','decomp','vars','est','mdl','nsource','ntarget'])['decision'].median()
    df_agg = df_agg.reset_index()
    df_agg_pivot = df_agg.pivot(index=['decomp'], columns='vars', values='decision')
    df_agg_pivot = df_agg_pivot.reset_index(drop=True)
    df_agg_pivot.columns.name = ''
    print(df_agg_pivot.columns)
    
    plt.figure(figsize=(5,1))
    ax = sns.heatmap(1 - df_agg_pivot, 
                    annot=df_agg_pivot, fmt='.2f', 
                    cmap=cmap, vmin=0, vmax=1,
                    yticklabels=False, linewidths=2,
                    #  cbar_kws={'label': 'pvalue'},
                    cbar = False,
        )
    ax.xaxis.tick_top()
    ax.set_title('Aggregate', pad=10)
    plt.savefig(result_file.replace('.csv','target_inference_agg.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_agg_pivot

def plot_detailed_simulation(df, decomp, result_file):
    df_detail = df[(df.decomp == decomp) & (df.est == 'onestep')]
    df_detail.loc[:, 'vars'] = df_detail.loc[:, 'vars'].apply(parse_proposed_vars)
    df_detail = df_detail[~df_detail.vars.str.contains('X1,X4|X2,X3|X1,X2|X3,X4')]
    df_detail = df_detail.groupby(by=['level','decomp','vars','est','mdl','nsource','ntarget'])['decision'].median()
    df_detail = df_detail.reset_index()
    df_detail = df_detail[['vars','decision']]
    df_detail.set_index('vars', inplace=True)
    df_detail.index.name = None
    # df_detail.columns = ['SHIFT']
    print(df_detail.index)

    plt.figure(figsize=(2,3))
    ax = sns.heatmap(df_detail, 
                    annot=1 - df_detail, fmt='.2f', 
                    cmap=cmap, vmin=0, vmax=1,
                    xticklabels=False,
                    yticklabels=decomp=='Cond_Outcome',
                    cbar=False
                    )
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('Detailed $\dagger$')
    plt.savefig(result_file.replace('.csv', '_detail.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_detail

def plot_comparators_simulation(comp_data, target_data, decomp, num_jobs, significance_level):
    num_p = pd.read_csv(target_data).shape[1] - 1  # number of features
    
    df = concat_files_to_df([comp_data], num_jobs)
    print(df)
    
    # Remove rows for CausalForestExplanation
    df = df[df['est'] != 'CausalForestExplanation']

    # Remove rows for DomainClassifierExplanation
    df = df[df['est'] != 'DomainClassifierExplanation']

    # Do Bonferroni correction on each est group
    df['significance_level'] = significance_level
    if BONFERRONI_CORRECTION:
        df['significance_level'] = df.groupby(['nsource','ntarget','est'])['significance_level'].transform(lambda x: x / len(x))

    # Get excluded vars for TEVIM and KCI tests, as the vars column logs complement of the variable of interest
    explanation_tests = df.est.str.contains('KCIOutcomeTest|TEVIMTest|KCICovariateTest')
    df.loc[explanation_tests, 'vars'] = df.loc[explanation_tests, 'vars'].transform(lambda x: parse_comp_vars(x, num_p))
    df = df[~df.vars.str.contains('X1,X4|X2,X3|X1,X2|X3,X4')]

    # Decide test by comparing to significance level
    # df_annot = df
    df.loc[:,'significant'] = df.loc[:,'pvalue'] < df.loc[:,'significance_level']
    df['significant'] = df['significant'].astype(float)
    df = df.groupby(by=['level','decomp','vars','est','mdl','nsource','ntarget'])['significant'].median()
    df = df.reset_index()

    # Pivot df on vars and est columns
    df_pivot = df.pivot(index=['vars','nsource','ntarget'], columns='est', values='significant')

    def trim_dataframe(df_pivot):
        # Sort by number of vars
        df_pivot = df_pivot.reset_index()
        df_pivot['vars_num'] = df_pivot['vars'].transform(lambda x: x.count(','))

        # Merge by var_idx in names and sort by vars_index
        df_pivot = df_pivot.sort_values(by=['nsource','vars_num'], ascending=True)

        df_pivot = df_pivot[df_pivot.nsource == 8000]

        # Drop all indices, group name, and vars columns
        df_pivot = df_pivot.drop(columns=['vars_num', 'nsource', 'ntarget'])  # assumes nsource == ntarget
        df_pivot.set_index('vars', inplace=True)
        df_pivot.index.name = None
        df_pivot.columns.name = None
        return df_pivot

    df_pivot = trim_dataframe(df_pivot)

    # Write to csv
    df_pivot.to_csv(comp_data.replace('.csv', '_pivot.csv'), index=False)
    print(df_pivot.index)

    explanation_tests_cols = df_pivot.columns.str.contains('KCIOutcomeTest|TEVIMTest|KCICovariateTest')  # fail to reject means variables are flagged
    df_pivot.loc[:, explanation_tests_cols] = 1 - df_pivot.loc[:, explanation_tests_cols]  # fail-to-reject rate

    df_pivot = df_pivot.rename(columns=COMPARATOR_NAMES)
    plt.figure(figsize=(9,3))
    ax = sns.heatmap(1 - df_pivot, 
                    annot=df_pivot, fmt='.2f',
                    cmap=cmap, vmin=0, vmax=1,
                    yticklabels=decomp=='Cond_Cov',
                    cbar=False
                    )
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('Baselines %s' % COMPARATOR_NAMES[decomp])
    plt.savefig(comp_data.replace('.csv', '_detail.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_pivot

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    np.random.seed(11563)

    # Get group names
    names = get_group_names(args.grouping_data)
    print("Group names", names)

    # Plot aggregate and detailed tests
    plot_aggregate(args.decomp_file, args.significance_level)
    if args.decomposition != 'Agg':
        plot_detailed(args.decomp_file, args.decomposition, names, args.significance_level)

    # Plot comparators
    if args.decomposition != 'Agg':
        plot_comparators_real(args.comparator_file, args.target_data_file, args.decomposition, names, args.significance_level)
        
    
if __name__ == "__main__":
    main()
