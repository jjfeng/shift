import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from decompositions import DecompositionEstimator
from common import read_csv, make_loss_func

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain performance differences across contexts"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed offset from job idx",
    )
    parser.add_argument(
        "--source-data-template",
        type=str,
        default="_output/source_dataJOB.csv",
    )
    parser.add_argument(
        "--target-data-template",
        type=str,
        default="_output/target_dataJOB.csv",
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlJOB.pkl",
    )
    parser.add_argument(
        "--out-file-template",
        type=str,
        default="_output/outJOB.csv",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
    )
    args = parser.parse_args()
    args.source_data = args.source_data_template.replace("JOB",
            str(args.job_idx))
    args.target_data = args.target_data_template.replace("JOB",
            str(args.job_idx))
    args.mdl_file = args.mdl_file_template.replace("JOB",
            str(args.job_idx))
    args.out_file = args.out_file_template.replace("JOB",
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

    sourceX, sourceY = read_csv(args.source_data)
    targetX, targetY = read_csv(args.target_data)

    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)
    
    mean_modeler = RandomForestRegressor(n_estimators=100, n_jobs=4)
    propensity_modeler = LogisticRegression()
    print(sourceX.shape)

    loss_func = make_loss_func("brier", ml_mdl)
    
    source_loss = loss_func(sourceX, sourceY).mean()
    target_loss = loss_func(targetX, targetY).mean()
    print("target loss", target_loss, "source loss", source_loss)

    decomposer = DecompositionEstimator(
        sourceX.to_numpy(),
        sourceY.to_numpy(),
        targetX.to_numpy(),
        targetY.to_numpy(),
        loss_func=loss_func,
        mean_modeler=mean_modeler,
        propensity_modeler=propensity_modeler)
    result_df, fold_result_df = decomposer.do_inference(np.array([True, False, False]))
    print(result_df)
    print(fold_result_df)

    result_df, fold_result_df = decomposer.do_inference(np.array([True, True, True]))
    print(result_df)
    print(fold_result_df)

if __name__ == "__main__":
    main()