"""
Code for computing aggregate and detailed tests, both test statistics and p-values
"""

import pickle
import logging
import argparse

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_loader import DataLoader
from estimate_datashifter import DetectorTestExplainer
from decomp_explainer import ExplainerInference
from common import read_csv, read_groupings, LossEvaluator, COND_OUTCOME_STR, COND_COV_STR
from common import get_bootstrap_diff_metric, get_pred_threshold_at_recall
from common import RevisedModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test performance differences in subgroups"
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
        "--is-oracle",
        action="store_true",
        help="whether to use an oracle model for the outcome model",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['outcome', 'accuracy', 'brier', 'balanced_accuracy', 'ppv'],
        help="loss function",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="",
        choices=["", COND_OUTCOME_STR, COND_COV_STR, f"{COND_COV_STR}+{COND_OUTCOME_STR}", "Agg"],
        help="type of detailed decomposition, either conditional covariate or outcome or or both",
    )
    parser.add_argument(
        "--filter-independent-features",
        action="store_true",
        help="whether to filter features independent of the loss function",
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
        "--source-data-generator",
        type=str,
        default="",
        help="source data",
    )
    parser.add_argument(
        "--target-data-generator",
        type=str,
        default="",
        help="target data",
    )
    parser.add_argument(
        "--grouping-data",
        type=str,
        default="_output/grouping_data.csv",
        help="grouping data",
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
        "--do-aggregate",
        action="store_true",
        help="compute aggregate decomposition",
    )
    parser.add_argument(
        "--do-grid-search",
        action="store_true",
        help="do grid search in outcome and density ratio modeling",
    )
    parser.add_argument(
        "--one-sided-test",
        action="store_true",
        help="create bootstrap intervals for one-sided test",
    )
    parser.add_argument(
        "--do-test-greater",
        action="store_true",
        help="test for alternate hypothesis that loss difference is greater than tolerance",
    )
    parser.add_argument(
        "--do-clipping",
        action="store_true",
        help="clip density ratios for ustatistics at a threshold",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0,
        help="param to threshold loss differences for non-perfect null tests",
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0,
        help="param to threshold size of detected subgroup",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="number of bins to bin expected outcome function for covariate shift test",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="significance level for critical values (e.g. 0.05)",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=10000,
        help="number of bootstrap samples to take for constructing rejection region by mulitplier bootstrap method",
    )
    parser.add_argument(
        "--combos",
        type=str,
        help="list of subset masks split by + sign, each mask separated by , sign",
    )
    parser.add_argument(
        "--gridsearch-polynom-lr",
        action="store_true",
        help="whether to add logistic regression with polynomial features in grid search",
    )
    parser.add_argument(
        "--save-detectors",
        action="store_true",
        help="save detectors for each test",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/resultestimateJOB.csv",
        help="results file to store value estimates, confidence intervals",
    )
    parser.add_argument(
        "--explainer-out-file",
        type=str,
        default="_output/explainerJOB.pkl",
        help="detectors output file",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
        help="log file",
    )
    args = parser.parse_args()
    assert args.num_bins > 1
    args.train_data = args.train_data_template.replace("JOB", "1")  # train data to compute class weight for balancing
    args.source_data = args.source_data_template.replace("JOB",
            str(args.job_idx))
    args.target_data = args.target_data_template.replace("JOB",
            str(args.job_idx))
    args.source_data_generator = args.source_data_generator.replace("JOB", str(args.job_idx)) if args.source_data_generator!="" else None
    args.target_data_generator = args.target_data_generator.replace("JOB", str(args.job_idx)) if args.target_data_generator!="" else None
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    args.explainer_out_file = args.explainer_out_file.replace("JOB",
            str(args.job_idx))
    args.decomposition = args.decomposition.split("+")
    if args.combos is not None:
        combos = args.combos.split("+")
        args.combos = [np.array(list(map(int, combo.split(","))), dtype=bool) for combo in combos]
        print(args.combos)
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
    logging.info("Loss %s CLASS WEIGHT for balanced accuracy %s", args.loss, class_weight)

    if args.target_data_generator is not None:
        with open(args.target_data_generator, "rb") as f:
            target_dg = pickle.load(f)
            print("TARGET X", target_dg.x_mean)
            print(target_dg.beta)
    else:
        target_dg = None
    if args.source_data_generator is not None:
        with open(args.source_data_generator, "rb") as f:
            source_dg = pickle.load(f)
            print("SOURCE X", source_dg.x_mean)
            print(source_dg.beta)
    else:
        source_dg = None

    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    result_ci = get_bootstrap_diff_metric(sourceY, targetY, ml_mdl.predict(sourceX), ml_mdl.predict(targetX),
                                              class_weight, args.loss, n_bootstrap=1000, alpha=0.05, rng_seed=0)
    logging.info("CI at 0.95 for %s difference, bootstrap %0.4f +- (%0.4f, %0.4f)", args.loss, result_ci[1], result_ci[0], result_ci[2])
    logging.info("Class prevalence source %s target %s", sourceY.mean(), targetY.mean())

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

    source_loader = DataLoader(sourceX[keep_indices_source], sourceY[keep_indices_source], vdict=variable_dict)
    target_loader = DataLoader(targetX[keep_indices_target], targetY[keep_indices_target], vdict=variable_dict)
    logging.info("NUM N source %d target %d, NUM P %d", source_loader.num_n, target_loader.num_n, source_loader.num_p)

    source_loss = loss_func(source_loader._get_X(), source_loader._get_Y()).mean()
    target_loss = loss_func(target_loader._get_X(), target_loader._get_Y()).mean()
    logging.info("TOTAL LOSS DIFF: %f - %f = %f", target_loss, source_loss, target_loss - source_loss)

    # Get aggregate decomposition
    shift_explainer = DetectorTestExplainer(
        source_loader=source_loader,
        target_loader=target_loader,
        source_data_generator=source_dg,
        target_data_generator=target_dg,
        loss_func=loss_func,
        ml_mdl=ml_mdl,
        do_grid_search=args.do_grid_search,
        do_clipping=args.do_clipping,
        gridsearch_polynom_lr=args.gridsearch_polynom_lr,
        is_oracle=args.is_oracle,
        tolerance=args.tolerance,
        prevalence=args.prevalence,
        num_bins=args.num_bins,
        test_greater=args.do_test_greater,
        filter_independent_features=args.filter_independent_features,
        filter_significance_level=args.significance_level,)  # TODO: allow different significance levels for filtering and tests

    if args.combos is not None:
        # Evaluate certain variable subsets
        explainer = ExplainerInference(
            num_obs=source_loader.num_n,
            num_p=source_loader.num_p,
            shift_explainer=shift_explainer,
            detailed_lst=args.decomposition,
            combos=args.combos,
            do_aggregate=args.do_aggregate,
            one_sided_test=args.one_sided_test,
            significance_level=args.significance_level,
            num_bootstrap=args.num_bootstrap
        )
    else:
        raise NotImplementedError
    explainer.run_tests()

    res_df = explainer.summary()
    print(res_df)
    logging.info(res_df)
    res_df["job"] = args.job_idx
    res_df["mdl"] = args.mdl_file
    res_df["nsource"] = source_loader.num_n
    res_df["ntarget"] = target_loader.num_n
    res_df.to_csv(args.result_file, index=False)

    if args.save_detectors:
        detectors = {
            "agg_detectors_x": explainer.shift_explainer.agg_detectors_x,
            "agg_detectors_y": explainer.shift_explainer.agg_detectors_y,
            "detail_detectors_x": explainer.shift_explainer.detail_detectors_x,
            "detail_detectors_y": explainer.shift_explainer.detail_detectors_y,
            "agg_correlated_features": explainer.shift_explainer.agg_correlated_features,
        }
        with open(args.explainer_out_file, 'wb') as f:
            pickle.dump(detectors, f)
            # pickle.dump(explainer, f)  # saves space

if __name__ == "__main__":
    main()
