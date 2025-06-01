import os
import logging
import argparse

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

METHOD_NAMES = {
    'onestep': 'SHIFT',
    'KCIOutcomeTestAgg': 'KCI',
    'MMDOutcomeTestAgg': 'MMD',
    'KCICovariateTestAgg': 'KCI',
    'MMDCovariateTestAgg': 'MMD',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="concatenate csvs"
    )
    parser.add_argument(
        "--result-files-estimate",
        type=str,
        help="output of estimate",
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
        "--plot-component",
        type=str,
        help="type of test to plot, X or Y",
        default="Y"
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="whether to have legend",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        help="number of jobs to read all estimates from data replicates",
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
    args.result_files_estimate = args.result_files_estimate.split(",")
    if args.result_files_comparators is not None:
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

def confidence_interval(data, alpha=0.05):
    bootstrap_mean = bootstrap(data, np.mean, confidence_level=1-alpha)
    return bootstrap_mean['confidence_interval']

def plot_rejection_vs_est(data, significance_level, plotfile, show_component=False):
    sns.set_context('notebook', font_scale=2)
    plt.clf()
    print("DATA AMOUNT", len(data))
    print(data)
    data["est"] = data["est"].replace(METHOD_NAMES)

    ax = sns.catplot(
        data=data,
        x="est",
        y="decision",
        kind="bar",
        legend=False,
        color='Red',
    )       
    ax.set(
        title=None,
        xlabel=None,
        ylabel='Power')
    
    ax.fig.tight_layout()
    plt.savefig(plotfile, bbox_inches="tight")
    print("test decision plot ", plotfile)

    # summary_df = data.groupby("est")["decision"].aggregate(lambda x: confidence_interval(x, alpha=0.05))

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )

    all_res_estimate = concat_files_to_df(args.result_files_estimate, args.num_jobs)
    all_res_estimate = all_res_estimate.reset_index(drop=True)
    print(all_res_estimate)

    if args.result_files_comparators is not None:
        all_res_comparators = concat_files_to_df(args.result_files_comparators, args.num_jobs)
        all_res_comparators = all_res_comparators.reset_index(drop=True)
        print(all_res_comparators)

        # Get test decision
        all_res_comparators['decision'] = all_res_comparators['pvalue'] <= args.significance_level
        print(all_res_comparators)

        # Concatenate results
        all_res_estimate = pd.concat([all_res_estimate, all_res_comparators], axis=0, join='outer')
        
    
    print(all_res_estimate)
    print("all_res_estimate.est", all_res_estimate.est.unique())
    print("all_res_estimate.vars", all_res_estimate.vars.unique())

    # Test to plot
    all_res_estimate = all_res_estimate[all_res_estimate.vars == args.plot_component]

    # Methods to plot
    all_res_estimate = all_res_estimate[all_res_estimate.est.isin([
        'onestep',
        'KCIOutcomeTestAgg',
        'MMDOutcomeTestAgg',
        'KCICovariateTestAgg',
        'MMDCovariateTestAgg',
    ])]

    all_res_estimate.to_csv(args.csv_file_estimate, index=False, na_rep=np.nan)
    
    summary_df = all_res_estimate[['level','decomp','vars','value','est','decision','nsource']].groupby(['level','decomp','vars', 'est', 'nsource']).mean()
    summary_df.to_csv(args.summary_csv_file)
    
    print("PLOT REJECTION RATE")
    if (args.coverage_agg_file is not None) and len(all_res_estimate[all_res_estimate.level == "agg"]):
        plot_rejection_vs_est(all_res_estimate, args.significance_level, args.coverage_agg_file)    

if __name__ == "__main__":
    main()
