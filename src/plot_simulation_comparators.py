import os
import logging
import argparse

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(
        description="concatenate csvs"
    )
    parser.add_argument(
        "--result-files-comparators",
        type=str,
        help="output of comparators",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        help="significance level of test, e.g. 0.05",
    )
    parser.add_argument(
        "--plot-components",
        type=str,
        help="list of aggregate, detailed decompositions, and comparators to plot, comma separated",
        default="agg,explained_ratio"
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="whether to have legend",
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
        help="feature mappings file e.g. X1 to feature name, csv file",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        help="number of jobs to read all estimates from data replicates",
    )
    parser.add_argument(
        "--scale-bias",
        help="scale the bias plot by root n",
        action="store_true",
    )
    parser.add_argument(
        "--one-sided-test",
        help="decide tests by one sided interval",
        action="store_true",
    )
    parser.add_argument(
        "--summary-csv-file",
        type=str,
        default="_output/summary.csv",
        help="output file of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
        help="log file",
    )
    parser.add_argument(
        "--csv-file-estimate",
        type=str,
        help="output file of estimates from all data replicates",
    )
    parser.add_argument(
        "--coverage-detail-file",
        type=str,
        help="coverage plot for detailed decompositions",
    )
    parser.add_argument(
        "--coverage-agg-file",
        type=str,
        help="coverage plot for aggregate decompositions",
    )
    args = parser.parse_args()
    args.plot_components = args.plot_components.split(",")
    args.result_files_comparators = args.result_files_comparators.split(",")
    return args

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

def plot_rejection_vs_n(data, significance_level, plotfile, show_component=False):
    sns.set_context('paper', font_scale=4)
    plt.clf()
    print("DATA AMOUNT", len(data))
    data["est"] = data["est"].str.replace("eif", "one-step correct")
    if show_component:
        data["decomp_component"] = data["decomp"] + data["component"]
    else:
        data["decomp_component"] = data["decomp"]

    facet_height = 5  # height of each facet in inches
    aspect_ratio = 1.5  # width = aspect_ratio * height
    ax = sns.relplot(
        data=data,
        x="nsource",
        y="rejected",
        hue="est",
        row="vars",
        kind="line",
        markers=True,
        linewidth=3,
        legend=True,
        height=facet_height,
        aspect=aspect_ratio,
        facet_kws={'sharex': True, 'sharey': True}
    )
    print("SIGNIFICANCE LEVEL", significance_level)
    custom_ticks = data.nsource.unique()
    ax.set(ylim=(0, 1), xticks=custom_ticks)
    ax.set_xticklabels(labels=custom_ticks, rotation=45)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    for axis in ax.axes.flat:
        axis.axhline(significance_level, ls="--", color="black")
        axis.axhline(1 - significance_level, ls="--", color="black")
        
    ax.set_titles(row_template="Subset = {row_name}, tolerance = 0.0") 
    
    # update to pretty axes and titles
    ax.set_axis_labels(
        "n",
        "Rejection Rate"
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.fig.tight_layout()
    plt.subplots_adjust(hspace=1)
    plt.savefig(plotfile, bbox_inches="tight")
    print("test decision plot ", plotfile)

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    feature_names_df = pd.read_csv(args.feature_names_file)


    all_res_estimate = concat_files_to_df(args.result_files_comparators, args.num_jobs)
    all_res_estimate = all_res_estimate.reset_index(drop=True)
    # Remove CausalForestExplanation as it does not output pvalues
    all_res_estimate = all_res_estimate[all_res_estimate['est']!='CausalForestExplanation']
    print(all_res_estimate)

    # Get test decision
    all_res_estimate['rejected'] = (all_res_estimate['pvalue'] <= args.significance_level)

    print("all_res_estimate.est", all_res_estimate.est.unique())

    all_res_estimate.to_csv(args.csv_file_estimate, index=False, na_rep=np.nan)
    
    summary_df = all_res_estimate[['level','decomp','vars','value','est','rejected','nsource']].groupby(['level','decomp','vars', 'est', 'nsource']).mean()
    summary_df.to_csv(args.summary_csv_file)
    
    print("PLOT REJECTION RATE")
    if (args.coverage_detail_file is not None) and len(all_res_estimate[all_res_estimate.level == "detail"]) and ('test_statistic' in args.plot_components):    
        plot_rejection_vs_n(all_res_estimate[all_res_estimate.level == "detail"], args.significance_level, args.coverage_detail_file)
    if (args.coverage_agg_file is not None) and len(all_res_estimate[all_res_estimate.level == "agg"]) and ('agg' in args.plot_components):
        plot_rejection_vs_n(all_res_estimate[all_res_estimate.level == "agg"], args.significance_level, args.coverage_agg_file)    

if __name__ == "__main__":
    main()
