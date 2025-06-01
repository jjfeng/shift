"""Get features important to detector and performances on detected groups
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data_loader import DataLoader
from comparators import *
from common import read_csv, LossEvaluator, read_groupings
from common import get_pred_threshold_at_recall
from common import get_bootstrap_metric

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
        "--explainer-file",
        type=str,
        help="fitted detector models which are to be explained",
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
    args.explainer_file = args.explainer_file.replace("JOB", 
            str(args.job_idx))
    if args.combos is not None:
        combos = args.combos.split("+")
        args.combos = [np.array(list(map(int, combo.split(","))), dtype=bool) for combo in combos]
        print(args.combos)
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    return args

def get_acc_auc_ppv(mdl, X, Y, threshold):
    pred_prob = mdl.predict_proba(X)[:,1]
    pred_Y = pred_prob > threshold
    assert len(Y)==len(pred_Y), "Y, pred_Y should be of same shape"
    conf_intervals = get_bootstrap_metric(Y, pred_Y, pred_prob, n_bootstrap=1000, alpha=0.05)  # format is (alpha/2, 0.5, 1-alpha/2) quartiles
    performances = {
        'acc': [(pred_Y == Y).mean()],
        'acc_lower': [conf_intervals['acc'][0]],
        'acc_upper': [conf_intervals['acc'][2]],
        'auc': [roc_auc_score(Y, pred_prob)],
        'auc_lower': [conf_intervals['auc'][0]],
        'auc_upper': [conf_intervals['auc'][2]],
        'ppv': [(pred_Y & Y).sum() / pred_Y.sum()],
        'ppv_lower': [conf_intervals['ppv'][0]],
        'ppv_upper': [conf_intervals['ppv'][2]],
    }  # each value is a list so it in can be converted to dataframe
    return performances

def get_detectors(explainer_data, data: DataLoader=None, feature_names=None):
    detector_x_fn, detector_y_fn = None, None
    tree_importance_x, tree_importance_y = None, None
    with open(explainer_data, 'rb') as f:
        detectors = pickle.load(f)
    correlated_features = detectors['agg_correlated_features']

    if len(detectors['agg_detectors_x']) > 0:
        detector_x, omega = detectors['agg_detectors_x'][0]
        def detector_x_fn(data_loader):
            data_loader = data_loader.copy()
            data_loader.subset_X(correlated_features)
            return detector_x.predict(data_loader, omega) > 0
        # detector_x_fn = lambda data_loader: detector_x.predict(data_loader, omega) > 0
        tree_importance_x = get_tree_importance_fitted(detector_x_fn, data)
        tree_importance_x = pd.DataFrame({'feature': feature_names, 'importance': tree_importance_x})
        tree_importance_x = tree_importance_x.sort_values(by='importance', ascending=False)
    if len(detectors['agg_detectors_y']) > 0:
        detector_y = detectors['agg_detectors_y'][0]  # Take only 1 detector
        tree_importance_y = detector_y.feature_importances_
        detector_y_fn = lambda data_loader: detector_y.predict(data_loader.X) > 0
        tree_importance_y = pd.DataFrame({'feature': feature_names, 'importance': tree_importance_y})
        tree_importance_y = tree_importance_y.sort_values(by='importance', ascending=False)
    return detector_x_fn, detector_y_fn, tree_importance_x, tree_importance_y

def get_tree_importance_fitted(detector_x, data: DataLoader):
    if data is None or len(np.unique(detector_x(data))) < 2:
        return None
    mdl = GradientBoostingClassifier(max_depth=2, n_estimators=200)
    mdl.fit(data.X, detector_x(data))
    return mdl.feature_importances_

def get_acc_detected(ml_mdl, detector, source_loader, target_loader, threshold=0.5):
    source_mask = detector(source_loader)
    target_mask = detector(target_loader)
    acc_detected_source = get_acc_auc_ppv(ml_mdl, source_loader.mdl_X[source_mask], source_loader.Y[source_mask], threshold)
    acc_detected_target = get_acc_auc_ppv(ml_mdl, target_loader.mdl_X[target_mask], target_loader.Y[target_mask], threshold)
    return acc_detected_source, acc_detected_target, source_mask, target_mask

def save_detected_data(output_file, detector, source_loader, target_loader, correlated_features):
    source_mask = detector(source_loader)
    target_mask = detector(target_loader)
    cols = ["C#X%d" % (i+1) for i in np.arange(sum(correlated_features))] + ["cD#detect"]
    
    df = pd.DataFrame(np.concatenate([
        target_loader.X[:, correlated_features],
        target_mask.astype(int).reshape(-1,1)
    ], axis=1), columns=cols)
    df.to_csv(output_file.replace(".csv", "_target.csv"), index=False)

    df = pd.DataFrame(np.concatenate([
        source_loader.X[:, correlated_features],
        source_mask.astype(int).reshape(-1,1)
    ], axis=1), columns=cols)
    df.to_csv(output_file.replace(".csv", "_source.csv"), index=False)
 
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
    feature_names = sourceX.columns
    sourceX, sourceY = sourceX.to_numpy(), sourceY.to_numpy()
    targetX, targetY = read_csv(args.target_data)
    targetX, targetY = targetX.to_numpy(), targetY.to_numpy()
    variable_dict = read_groupings(args.grouping_data)
    class_weight = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=trainY)

    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    if args.choose_threshold:
        chosen_threshold = get_pred_threshold_at_recall(
            sourceY, ml_mdl.predict_proba(sourceX)[:,1], required_recall=0.3
        )
    else:
        chosen_threshold = 0.5
    print("Chosen threshold %.3f, by recall around 0.3: %s" % (chosen_threshold, args.choose_threshold))
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

    with open(args.explainer_file, 'rb') as f:
        detectors = pickle.load(f)
    correlated_features = detectors['agg_correlated_features']

    detector_x, detector_y, tree_importance_x, tree_importance_y = get_detectors(args.explainer_file, target_loader, feature_names)

    if detector_x is not None:
        acc_detected_source, acc_detected_target, source_mask, target_mask = get_acc_detected(ml_mdl, detector_x, source_loader, target_loader)
        save_detected_data(
            args.result_file.replace(".csv", "_detected_data_covariate.csv"),
            detector_x,
            source_loader,
            target_loader,
            correlated_features
        )
        logging.info("\nCovariate: detected group has %d (%.3f) points in Source \n%s", sum(source_mask), np.mean(source_mask), pd.DataFrame(acc_detected_source))
        with open("masks.pkl", "wb") as f:
            pickle.dump({'source': source_mask, 'target': target_mask}, f)
        logging.info("Source Acc: %.3f (%.3f, %.3f), PPV: %.3f (%.3f, %.3f)", acc_detected_source['acc'][0], acc_detected_source['acc_lower'][0], acc_detected_source['acc_upper'][0], acc_detected_source['ppv'][0], acc_detected_source['ppv_lower'][0], acc_detected_source['ppv_upper'][0])
        logging.info("\nCovariate: detected group has %d (%.3f) points in Target \n%s", sum(target_mask), np.mean(target_mask), pd.DataFrame(acc_detected_target))
        logging.info("Target Acc: %.3f (%.3f, %.3f), PPV: %.3f (%.3f, %.3f)", acc_detected_target['acc'][0], acc_detected_target['acc_lower'][0], acc_detected_target['acc_upper'][0], acc_detected_target['ppv'][0], acc_detected_target['ppv_lower'][0], acc_detected_target['ppv_upper'][0])
        logging.info("\nCovariate detector top 5 important \n%s", tree_importance_x[:5].to_string())

    if detector_y is not None:
        acc_detected_source, acc_detected_target, source_mask, target_mask = get_acc_detected(ml_mdl, detector_y, source_loader, target_loader)
        save_detected_data(
            args.result_file.replace(".csv", "_detected_data_outcome.csv"),
            detector_y,
            source_loader,
            target_loader,
            correlated_features=np.ones((source_loader.X.shape[1],), dtype=bool)
        )
        logging.info("\nOutcome: detected group %d (%.3f) has acc in Source \n%s", sum(source_mask), np.mean(source_mask), pd.DataFrame(acc_detected_source))
        logging.info("Source Acc: %.3f (%.3f, %.3f), PPV: %.3f (%.3f, %.3f)", acc_detected_source['acc'][0], acc_detected_source['acc_lower'][0], acc_detected_source['acc_upper'][0], acc_detected_source['ppv'][0], acc_detected_source['ppv_lower'][0], acc_detected_source['ppv_upper'][0])
        logging.info("\nOutcome: detected group %d (%.3f) has acc in Target \n%s", sum(target_mask), np.mean(target_mask), pd.DataFrame(acc_detected_target))
        logging.info("Target Acc: %.3f (%.3f, %.3f), PPV: %.3f (%.3f, %.3f)", acc_detected_target['acc'][0], acc_detected_target['acc_lower'][0], acc_detected_target['acc_upper'][0], acc_detected_target['ppv'][0], acc_detected_target['ppv_lower'][0], acc_detected_target['ppv_upper'][0])
        logging.info("\nOutcome detector top 5 importance \n%s", tree_importance_y[:5].to_string())

    pd.concat([tree_importance_x, tree_importance_y], axis=0).to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main()