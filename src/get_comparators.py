"""
Run comparators on data files
"""

import os
import pickle
import logging
import argparse
import itertools
import scipy
import csv

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_loader import DataLoader
from comparators import *
from common import read_csv, LossEvaluator, read_groupings
from common import get_pred_threshold_at_recall

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain performance differences across contexts"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx for data replicates",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed offset from job idx",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov', 'Agg'],
        help="type of detailed decomposition for comparators, outcome or covariate",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['outcome', 'accuracy', 'brier', 'balanced_accuracy', 'ppv'],
        help="loss function",
    )
    parser.add_argument(
        "--do-grid-search",
        action="store_true",
        help="do grid search in outcome and density ratio modeling"
    )
    parser.add_argument(
        "--train-data-template",
        type=str,
        default="_output/train_dataJOB.csv",
        help="train data",
    )
    parser.add_argument(
        "--source-data-template",
        type=str,
        default="_output/source_dataJOB.csv",
        help="source data",
    )
    parser.add_argument(
        "--target-data-template",
        type=str,
        default="_output/target_dataJOB.csv",
        help="target data",
    )
    parser.add_argument(
        "--grouping-data",
        type=str,
        default="_output/grouping_data.csv",
        help="grouping data",
    )
    parser.add_argument(
        "--source-data-generator",
        type=str,
        help="source data generator in a pickle file",
    )
    parser.add_argument(
        "--target-data-generator",
        type=str,
        help="target data generator in a pickle file",
    )
    parser.add_argument(
        "--combos",
        type=str,
        help="list of subset masks split by + sign, each mask separated by , sign",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        help="significance level i.e. alpha for confidence intervals e.g. 0.05",
    )
    parser.add_argument(
        "--mdl-file",
        type=str,
        default="_output/mdl.pkl",
        help="trained model which is to be explained",
    )
    parser.add_argument(
        "--choose-threshold",
        action="store_true",
        help="choose threshold for prediction by desired recall",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/resultcomparatorJOB.csv",
        help="results file to store importances from comparators",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logcomparatorJOB.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.train_data = args.train_data_template.replace("JOB", "1")
    args.source_data = args.source_data_template.replace("JOB",
            str(args.job_idx))
    args.target_data = args.target_data_template.replace("JOB",
            str(args.job_idx))
    args.source_data_generator = args.source_data_generator.replace("JOB", 
            str(args.job_idx)) if args.source_data_generator is not None else None
    args.target_data_generator = args.target_data_generator.replace("JOB", 
            str(args.job_idx)) if args.target_data_generator is not None else None
    if args.combos is not None:
        combos = args.combos.split("+")
        args.combos = [np.array(list(map(int, combo.split(","))), dtype=bool) for combo in combos]
        print(args.combos)
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    
    _, trainY = read_csv(args.train_data)
    trainY = trainY.to_numpy()
    sourceX, sourceY = read_csv(args.source_data)
    sourceX, sourceY = sourceX.to_numpy(), sourceY.to_numpy()
    targetX, targetY = read_csv(args.target_data)
    targetX, targetY = targetX.to_numpy(), targetY.to_numpy()
    variable_dict = read_groupings(args.grouping_data)
    class_weight = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=trainY)

    target_dg = None
    source_dg = None
    if args.target_data_generator:
        with open(args.target_data_generator, "rb") as f:
            target_dg = pickle.load(f)
            print("TARGET X", target_dg.x_mean)
            print(target_dg.beta)
        with open(args.source_data_generator, "rb") as f:
            source_dg = pickle.load(f)
            print("SOURCE X", source_dg.x_mean)
            print(source_dg.beta)
    
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    if args.choose_threshold:
        chosen_threshold = get_pred_threshold_at_recall(
            sourceY, ml_mdl.predict_proba(sourceX)[:,1], required_recall=0.3
        )
    else:
        chosen_threshold = 0.5

    logging.info("Chosen threshold %.3f, by recall around 0.3: %s", chosen_threshold, args.choose_threshold)
    loss_func = LossEvaluator(args.loss, ml_mdl, class_weight=class_weight, threshold=chosen_threshold).get_loss

    if args.loss == 'ppv':
        # Keep only rows that are predicted to be positive
        keep_indices_source = ml_mdl.predict_proba(sourceX)[:,1] > chosen_threshold
        keep_indices_target = ml_mdl.predict_proba(targetX)[:,1] > chosen_threshold
    else:
        keep_indices_source, keep_indices_target = np.ones(sourceX.shape[0], dtype=bool), np.ones(targetX.shape[0], dtype=bool)

    source_loader = DataLoader(sourceX[keep_indices_source], sourceY[keep_indices_source], variable_dict)
    target_loader = DataLoader(targetX[keep_indices_target], targetY[keep_indices_target], variable_dict)
    logging.info("NUM N source %d target %d, NUM P %d", source_loader.num_n, target_loader.num_n, source_loader.num_p)
    
    source_loss = loss_func(source_loader._get_X(), source_loader._get_Y()).mean()
    target_loss = loss_func(target_loader._get_X(), target_loader._get_Y()).mean()
    logging.info("TOTAL LOSS DIFF: %f - %f = %f", target_loss, source_loss, target_loss - source_loss)

    if args.decomposition == "Cond_Cov":
        COMPARATORS = [
            KSTest,
            DomainClassifierExplanation,
            ScoreMethod,
            KCICovariateTest,
            # LinearMediationTest,
        ]
    elif args.decomposition == "Cond_Outcome":
        COMPARATORS = [
            ParametricChangeExplanation,
            ParametricAccExplanation,
            # RandomForestAccExplanation,
            # GBTAccExplanation,
            # CausalForestExplanation,
            KCIOutcomeTest,
            TEVIMTest,
        ]
    else:
        COMPARATORS = [
            KCIOutcomeTestAgg,
            KCICovariateTestAgg,
            # MMDOutcomeTestAgg,
            # MMDCovariateTestAgg,
        ]

    # Get comparator explanations
    res_df = None
    for Comparator in COMPARATORS:
        print("COMPARATOR", Comparator)
        explainer = Comparator(
            ml_mdl=ml_mdl,
            source_data_loader=source_loader,
            target_data_loader=target_loader,
            source_generator=source_dg,
            target_generator=target_dg,
            combos=args.combos,
            loss_func=loss_func,
            significance_level=args.significance_level,
        )
        explainer.do_decomposition()

        res_df = pd.concat(
            [res_df, explainer.summary()]
        )

    if res_df is not None:
        res_df["job"] = args.job_idx
        res_df["mdl"] = args.mdl_file
        res_df["nsource"] = source_loader.num_n
        res_df["ntarget"] = target_loader.num_n
        res_df.to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main()
