import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from data_generator import DataGenerator, DataGeneratorMultiNorm, DataGeneratorSeqNorm, DataGeneratorMultiNormAntiCausal
from data_generator import DataGeneratorMultiNormRestrict
from data_generator import DataGeneratorBox, DataGeneratorTree

def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for testing")
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
        help="random seed offset",
    )
    parser.add_argument(
        "--x-scale",
        type=str,
        default='2,2,2,2',
        help="scale of normal distributed additive error in independent normal and sequentially generated x-dist setting",
    )
    parser.add_argument(
        "--x-source-scale",
        type=str,
        default='2,2,2,2',
        help="scale of normal distributed additive error in independent normal and sequentially generated x-dist setting",
    )
    parser.add_argument(
        "--x-dist",
        type=str,
        default="unif",
        choices=["unif", "norm", "seq_norm", "restrictnorm", "anticausal", "tree", "box"],
        help="type of distribution for x",
    )
    parser.add_argument(
        "--x-mean",
        type=str,
        help="x mean in target, shifted for a subgroup or for all",
    )
    parser.add_argument(
        "--x-source-mean",
        type=str,
        help="x mean in source",
    )
    parser.add_argument(
        "--beta",
        type=str,
        help="comma separated list of coefficients for y logit"
    )
    parser.add_argument(
        "--source-beta",
        type=str,
        help="comma separated list of coefficients for y logit, needed to keep same distribution except in subgroup"
    )
    parser.add_argument(
        "--subgroupshift",
        action="store_true",
        help="shift outcome function or covariate distribution in subgroup defined by first column"
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="introduce non-linearity in outcome function by taking absolute value of first feature before computing outcome probability"
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        default=100,
        help="number of observations",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/data_logJOB.txt",
        help="log file",
    )
    parser.add_argument(
        "--out-data-gen-file",
        type=str,
        default="_output/datagenJOB.pkl",
        help="output data generator file in pickle format",
    )
    parser.add_argument(
        "--out-file-template",
        type=str,
        default="_output/dataJOB.csv",
        help="output data file in csv format",
    )
    args = parser.parse_args()
    args.beta = np.array(list(map(float, args.beta.split(","))))
    args.x_mean = np.array(list(map(float, args.x_mean.split(","))))
    args.x_scale = np.array(list(map(float, args.x_scale.split(","))))
    if args.source_beta is not None:
        args.source_beta = np.array(list(map(float, args.source_beta.split(","))))
    if args.x_source_mean is not None:
        args.x_source_mean = np.array(list(map(float, args.x_source_mean.split(","))))
    if args.x_source_scale is not None:
        args.x_source_scale = np.array(list(map(float, args.x_source_scale.split(","))))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    args.out_file = args.out_file_template.replace("JOB",
            str(args.job_idx))
    args.out_data_gen_file = args.out_data_gen_file.replace("JOB",
            str(args.job_idx))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    if args.x_dist == "unif":
        dg = DataGenerator(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear, scale=args.x_scale)
    elif args.x_dist == "norm":
        dg = DataGeneratorMultiNorm(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear, scale=args.x_scale)
    elif args.x_dist == "restrictnorm":
        dg = DataGeneratorMultiNormRestrict(beta = args.beta, source_beta=args.source_beta, intercept=0, x_mean=args.x_mean, x_source_mean=args.x_source_mean, nonlinear=args.nonlinear, subgroupshift=args.subgroupshift, scale=args.x_scale, source_scale=args.x_source_scale)
    elif args.x_dist == "seq_norm":
        dg = DataGeneratorSeqNorm(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear, scale=args.x_scale)
    elif args.x_dist == "anticausal":
        dg = DataGeneratorMultiNormAntiCausal(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear)
    elif args.x_dist == "box":
        dg = DataGeneratorBox(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear)
    elif args.x_dist == "tree":
        dg = DataGeneratorTree(beta = args.beta, intercept=0, x_mean=args.x_mean, nonlinear=args.nonlinear)

    X, y = dg.generate(args.num_obs)
    df = pd.DataFrame(X)
    df["y"] = y
    df.to_csv(args.out_file, index=False)

    if args.out_data_gen_file:
        with open(args.out_data_gen_file, "wb") as f:
            pickle.dump(dg, f)

if __name__ == "__main__":
    main()
