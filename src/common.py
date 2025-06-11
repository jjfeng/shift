"""
Utility functions such as for calculating risk, binning, model fitting
"""
import logging
import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve

import torch
from scipy.spatial import distance

COND_OUTCOME_STR = "Cond_Outcome"
COND_COV_STR = "Cond_Cov"
# Default hyperparams
N_ESTIMATORS = 400
DEFAULT_DEPTH = 16  # 8
DEFAULT_MAX_FEATURES = None
CV = 3
kernel_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LogisticRegression(penalty='l1', fit_intercept=True, solver='saga', max_iter=20000))])

class Binning:
    """
    Class to bin source probabilities mu_0 for outcome shifts
    """
    def __init__(self, init_prob, num_bins, type='space', eps=1e-10):
        self.do_binning = (num_bins > 0)
        self.bins = None
        self.centers = None
        self.num_bins = num_bins
        if self.do_binning:
            if type=='quantile':
                bins = np.quantile(init_prob, q=np.linspace(0,1,num_bins+1))
            elif type=='space':
                bins = np.linspace(min(init_prob)-eps,max(init_prob)+eps,num_bins+1)  # equally spaced bin intervals between min and max probs
            else:
                raise NotImplementedError
            centers = (bins[1:]+bins[:-1])/2
            max_prob = max(max(bins),max(init_prob))
            min_prob = min(min(bins),min(init_prob))
            centers = np.insert(np.insert(centers,len(centers),max_prob),0,min_prob)
            self.bins = bins
            self.centers = centers
    def bin(self, prob):
        if self.do_binning:
            ind = np.digitize(prob, self.bins, right=True)
            assert (np.min(ind) >= 0) and (np.max(ind) <= (self.num_bins+1)), "bins %s centers %s min ind %f max ind %f min center %f max center %f" % (self.bins,self.centers,np.min(ind),np.max(ind),np.min(self.centers[ind]),np.max(self.centers[ind]))
            return self.centers[ind]
        else:
            return prob  # return prob as-it-is for non-positive values of num_bins

class LossEvaluator:
    def __init__(self, loss_name: str, ml_mdl: BaseEstimator, class_weight: np.ndarray, threshold: float=0.5):
        self.loss_name = loss_name
        self.ml_mdl = ml_mdl
        self.class_weight = class_weight
        self.threshold = threshold
    
    def get_loss(self, x, y):
        if self.loss_name == "outcome":
            return y.flatten()
        elif self.loss_name == "brier":
            return np.power(self.ml_mdl.predict_proba(x)[:,1] - y.flatten(), 2)
        elif self.loss_name in ["accuracy", "ppv"]:
            pred_val = (self.ml_mdl.predict_proba(x)[:,1] > self.threshold).astype(int)
            return (pred_val != y.flatten()).astype(float)
        elif self.loss_name == "balanced_accuracy":
            weight = np.ones_like(y.flatten(), dtype=float)
            weight[y==1] = self.class_weight[1]  # TODO: pass in parameters
            weight[y!=1] = self.class_weight[0]
            print("LOSS WEIGHT", weight)
            pred_val = (self.ml_mdl.predict_proba(x)[:,1] > 0.5).astype(int)
            print("PRED TRUE Y", pred_val, y.flatten())
            return (pred_val != y.flatten()).astype(int) * weight

def get_pred_threshold_at_recall(y_true, y_proba, required_recall):
    _, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Find index closest to required recall
    i = np.argmin(np.abs(recall - required_recall))
    return thresholds[i]
        
class PipelineWeightedFit(Pipeline):
    """Extends Pipeline class in sklearn to pass sample_weight param during fit
    """
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            # NOTE: assumes classifier step in pipeline is called clf
            return super().fit(X, y, **{'clf__sample_weight': sample_weight})
        else:
            return super().fit(X, y)

class RevisedModel:
    """
    Class to store revised model in model fix experiment
    """
    def __init__(self, org_mdl, revised_mdl, vars_mask):
        self.org_mdl = org_mdl
        self.revised_mdl = revised_mdl
        self.vars_mask = vars_mask
    def _get_feats(self, X):
        """
        Revised model takes original model predictions and top k features
        """
        org_mdl_prob = self.org_mdl.predict_proba(X)[:,1:]
        return np.concatenate([
            np.log(org_mdl_prob/(1 - org_mdl_prob)),
            X[:, self.vars_mask],
        ], axis=1)
    def predict_proba(self, X):
        retest_feats = self._get_feats(X)
        return self.revised_mdl.predict_proba(retest_feats)
    def predict(self, X):
        retest_feats = self._get_feats(X)
        return self.revised_mdl.predict(retest_feats)

def to_safe_prob(prob, eps=1e-10):
    """
    Clip probs, used for density ratios
    """
    return np.maximum(eps, np.minimum(1 - eps, prob))

def convert_logit_to_prob(logit):
    return 1/(1 + np.exp(-logit))

def convert_prob_to_logit(prob, eps=1e-10):
    safe_prob = to_safe_prob(prob, eps)
    return np.log(safe_prob/(1 - safe_prob))

def get_complementary_logit(logit):
    return convert_prob_to_logit(1 - convert_logit_to_prob(logit))

def get_sigmoid_deriv(logit):
    p = convert_logit_to_prob(logit)
    return p * (1 - p)

def get_inv_sigmoid_deriv(prob):
    return 1/prob + 1/(1 - prob)

def get_density_ratio_from_classifier(pred_probs, scale=1, eps=1e-10):
    pred_prob1 = to_safe_prob(pred_probs[:,1], eps)
    return pred_prob1/(1 - pred_prob1) * scale

def read_csv(csv_file: str):
    """Read train, source, and target data files
    NOTE: last column is assumed to be label and to be binary 0, 1,
    NOTE: all features are assumed to be numeric
    """
    df = pd.read_csv(csv_file)
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    try:
        classes = np.unique(Y.astype(int))
        if (len(classes)!=2) or (set(classes)!=set([0,1])):
            raise NotImplementedError
    except:
        raise NotImplementedError
    X = X.astype(float)
    Y = Y.astype(int)
    print("X", X.head(5))
    print("Y", Y.head(5))
    return X, Y

def read_groupings(csv_file: str):
    '''Takes in groupings file and outputs a dictionary from variable index to its group
    NOTE: first column is variable index and last column is its group
    '''
    df = pd.read_csv(csv_file, encoding='utf-8')
    variable_number = df.iloc[:,0].astype(int) # first column
    variable_group = df.iloc[:,-1].astype(int) # last column
    variable_dict = dict(zip(variable_number, variable_group)) 
    return variable_dict

def get_n_jobs():
    n_cpu = int(os.getenv('OMP_NUM_THREADS')) if os.getenv('OMP_NUM_THREADS') is not None else 0
    n_jobs = max(n_cpu - 1, 1) if n_cpu > 0 else -1
    logging.info("NUM JOBS %d", n_jobs)
    return n_jobs

def compute_risk(loss_func, X, proba):
    """
    Computes model risk as p(y=1|x)loss(1,f(x)) + p(y=0|x)loss(0,f(x))
    """
    loss_Y1 = loss_func(X, np.ones(X.shape[0]))
    loss_Y0 = loss_func(X, np.zeros(X.shape[0]))
    assert loss_Y1.shape == proba.shape
    assert loss_Y0.shape == proba.shape
    return loss_Y1 * proba + loss_Y0 * (1 - proba)

def get_bootstrap_metric(Y_true, Y_pred, Y_prob, n_bootstrap=1000, alpha=0.05, rng_seed=0):
    """
    Bootstrap estimate of accuracy and AUC
    """
    rng = np.random.RandomState(rng_seed)
    indices = rng.randint(0, len(Y_true), (n_bootstrap, len(Y_true)))
    Y_true_boot = Y_true[indices]
    Y_pred_boot = Y_pred[indices]
    Y_prob_boot = Y_prob[indices]
    
    acc = np.mean(Y_true_boot == Y_pred_boot, axis=1)
    auc = np.array([roc_auc_score(Y_true_boot[i], Y_prob_boot[i]) if len(np.unique(Y_true_boot[i])) >= 2 else np.nan for i in range(n_bootstrap)])
    ppv = np.sum(Y_true_boot & Y_pred_boot, axis=1) / np.sum(Y_pred_boot, axis=1)
    
    results = {
        'acc': np.quantile(acc, q=[alpha/2, 0.5, 1-alpha/2]),
        'auc': np.quantile(auc, q=[alpha/2, 0.5, 1-alpha/2]),
        'ppv': np.quantile(ppv, q=[alpha/2, 0.5, 1-alpha/2]),
    }
    return results

def get_bootstrap_diff_metric(Y_true1, Y_true2, Y_pred1, Y_pred2, class_weight, metric_name, n_bootstrap=100, alpha=0.05, rng_seed=0):
    """
    Boostrap estimate of difference in accuracy of model on two datasets
    """
    weight1 = np.ones_like(Y_true1.flatten(), dtype=float)
    weight2 = np.ones_like(Y_true2.flatten(), dtype=float)
    if metric_name=="balanced_accuracy":
        weight1[Y_true1==1] = class_weight[1]
        weight1[Y_true1!=1] = class_weight[0]
        weight2[Y_true2==1] = class_weight[1]
        weight2[Y_true2!=1] = class_weight[0]
    rng = np.random.RandomState(rng_seed)
    results = []
    for _ in range(n_bootstrap):
        random_indices1 = rng.randint(0, len(Y_true1), len(Y_true1))
        m1 = np.mean((Y_true1[random_indices1] == Y_pred1[random_indices1]).astype(int) * weight1[random_indices1])

        random_indices2 = rng.randint(0, len(Y_true2), len(Y_true2))
        m2 = np.mean((Y_true2[random_indices2] == Y_pred2[random_indices2]).astype(int) * weight2[random_indices2])
        results.append(m2 - m1)
    lower_ci, median, upper_ci = np.quantile(results, q=[alpha/2, 0.5, 1-alpha/2])
    return lower_ci, median, upper_ci

def get_multiplier_bootstrap_dist(ics, masks, n_bootstrap=10000, rng_seed=0):
    """
    Gaussian multiplier bootstrap as done in 
    Yu‐Chin Hsu, Consistent tests for conditional treatment effects, The Econometrics Journal
    Centers the influence function before multiplying with standard normals
    """
    n_datapoints = ics[0].shape[0]
    rng = np.random.RandomState(rng_seed)
    assert ics[0].ndim==1, "test statistic should be a 1d array"
    multipliers = rng.normal(0, 1, size=(n_datapoints, n_bootstrap))
    estim_per_mdl = [
        np.mean(np.multiply(
            multipliers[mask], np.repeat((ic[mask]-np.mean(ic[mask])).reshape(-1,1), n_bootstrap, axis=1)
            ), axis=0)
        for ic, mask in zip(ics, masks) if mask.sum() > 0
    ]
    if len(estim_per_mdl) == 0:
        return np.zeros(n_datapoints)
    max_estim = np.max(
        np.r_[estim_per_mdl], axis=0
    )  # max across detectors
    return max_estim
    
def test_mmd(X_tr, X_te, alpha=0.05):
    assert len(X_tr.shape) == len(X_te.shape), "X_tr and X_te should have the same dimensions"
    if len(X_tr.shape) == 1:
        X_tr = X_tr.reshape(-1,1)
        X_te = X_te.reshape(-1,1)
    
    # MMD with fixed bandwidth using median heuristic
    from torch_two_sample import MMDStatistic
    mmd_test = MMDStatistic(len(X_tr), len(X_te))

    # torch_two_sample somehow wants the inputs to be explicitly casted to float 32.
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    all_dist = distance.cdist(X_tr, X_te, 'euclidean')
    median_dist = np.median(all_dist)
    t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
                        torch.autograd.Variable(torch.tensor(X_te)),
                        alphas=[1/median_dist], ret_matrix=True
                    )
    pvalue = mmd_test.pval(matrix)
    decision = pvalue < alpha
    return decision, pvalue, t_val.item()

def get_correlated_features(X, Y, alpha=0.05):
    """
    Get features that are correlated with the outcome using an MMD test for independence
    Returns binary vector indicating correlated features and
    pvalues that are used for tests
    """
    correlated_features_mask = np.zeros(X.shape[1], dtype=bool)
    pvalues = []
    for i in np.arange(X.shape[1]):
        decision, pval, _ = test_mmd(X[:,i], Y, alpha=alpha)
        correlated_features_mask[i] = decision
        pvalues.append(pval)
    return correlated_features_mask, pvalues

def get_twosided_threshold(values, tolerance):
    values = np.array(values)
    results = values.copy()
    positive_indices = (values>0)
    results[positive_indices] = np.minimum(values[positive_indices], tolerance)
    results[~positive_indices] = np.maximum(values[~positive_indices], -tolerance)
    return results
    
def get_density_model(max_feats:list[int], do_grid_search=False, model_args=None, gridsearch_polynom_lr: bool = False):
    """
    Probabilistic classifier for the density ratio models
    """
    if model_args is None:
        model_args = {'max_depth': DEFAULT_DEPTH, 'max_features': DEFAULT_MAX_FEATURES, 'n_estimators': N_ESTIMATORS}
    if do_grid_search:
        pipeline = PipelineWeightedFit([
            ('clf', RandomForestClassifier())
        ])  # a dummy classifier type is needed
        parameters = [
            {
                'clf': (CalibratedClassifierCV(RandomForestClassifier(
                    max_depth=model_args['max_depth'],
                    criterion='log_loss',
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    n_jobs=get_n_jobs(),
                    random_state=0), cv=CV),),
                'clf__estimator__max_depth': [2,4,6,8,16],
                'clf__estimator__max_features': max_feats
            },
            {
                'clf': (LogisticRegression(penalty=None, solver='saga', max_iter=1000),),
            },
            {
                'clf': (LogisticRegression(penalty='l2', max_iter=1000),),
                'clf__C': [1e-5,1e-04,0.001,0.1,1,10,100,1000],
            }
        ]
        if gridsearch_polynom_lr:
            kernel_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LogisticRegression(penalty='l2', fit_intercept=True, max_iter=20000, solver="saga"))])
            parameters += [{
                'clf': [kernel_model],
                'clf__linear__penalty': ['l1', 'l2'],
                'clf__linear__C': [100,10,1,0.1],
            }]
        model = GridSearchCV(pipeline, parameters, n_jobs=1, cv=CV, scoring="neg_log_loss", verbose=2)
    else:
        model = CalibratedClassifierCV(RandomForestClassifier(
            max_depth=model_args['max_depth'],
            max_features=model_args['max_features'],
            n_estimators=model_args['n_estimators'],
            n_jobs=-1,
            random_state=0,
            ), cv=CV)
        # model = LogisticRegression(penalty="l2")
    print("density model", model)
    return model

def get_outcome_model(
        is_binary: bool,
        max_feats: list[int],
        model_args=None,
        do_grid_search=False,
        n_jobs:int=-1,
        is_oracle: bool = False,
        gridsearch_polynom_lr: bool = False):
    if model_args is None:
        model_args = {'max_depth': DEFAULT_DEPTH, 'max_features': DEFAULT_MAX_FEATURES, 'n_estimators': N_ESTIMATORS, 'bootstrap': True}
    if is_binary:
        if do_grid_search:
            pipeline = PipelineWeightedFit([
                ('clf', RandomForestClassifier())
            ])
            parameters = []
            if max_feats is None:
                parameters = [
                    {
                        'clf': (LogisticRegression(penalty='l1', solver='saga', max_iter=1000),),
                        'clf__C': [0.001,0.1,1,10,100,1000],
                    },
                    {
                        'clf': (LogisticRegression(penalty='l2', max_iter=1000),),
                        'clf__C': [1e-5,1e-04,0.001,0.1,1,10,100,1000],
                    }
                    ]
            else:
                parameters = [
                {
                    'clf': (RandomForestClassifier(
                        max_depth=model_args['max_depth'],
                        criterion="log_loss",
                        max_features=model_args['max_features'],
                        n_estimators=model_args['n_estimators'],
                        n_jobs=get_n_jobs(),
                        random_state=0,
                        bootstrap=model_args['bootstrap']),),
                    'clf__max_depth': [2,4,6,8,16],
                    'clf__max_features': max_feats
                },
                {
                    'clf': (GradientBoostingClassifier(
                        # init=LogisticRegression(penalty=None, max_iter=2000),
                        max_depth=model_args['max_depth'],
                        n_estimators=model_args['n_estimators']),),
                    'clf__max_depth': [2,4]
                },
                {
                    'clf': (LogisticRegression(penalty=None, max_iter=1000),),
                },
                {
                    'clf': (LogisticRegression(penalty='l2', max_iter=1000),),
                    'clf__C': [1e-5,1e-04,0.001,0.1,1,10,100,1000],
                }
            ]
            if is_oracle:
                print("ORACLE")
                parameters += [
                    {
                        'clf': (LogisticRegression(penalty=None),)
                    }
                ]
            if gridsearch_polynom_lr:
                kernel_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LogisticRegression(penalty="l2", solver='saga', max_iter=20000))])
                parameters += [{
                    'clf': [kernel_model],
                    'clf__linear__penalty': ['l1', 'l2'],
                    'clf__linear__C': [100,10,1,0.1,0.001]
                }]
            model = GridSearchCV(pipeline, parameters, scoring='neg_log_loss',
                    cv=CV, n_jobs=1, verbose=2)
        else:
            if is_oracle:
                model = LogisticRegression(penalty=None)
            else:
                model = RandomForestClassifier(
                    criterion="log_loss",
                    max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    bootstrap=model_args['bootstrap'],
                    n_jobs=get_n_jobs(),
                    random_state=0
                )
            # model = LogisticRegression(penalty="l2")
    else:
        if do_grid_search:
            pipeline = PipelineWeightedFit([
                ('clf', RandomForestRegressor())
            ])
            kernel_model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', Ridge(alpha=1, max_iter=20000))])
            
            parameters = [{
                'clf': (RandomForestRegressor(
                    max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    n_jobs=get_n_jobs(),
                    random_state=0),),
                'clf__max_depth': [2,4,6,8,16],
                'clf__max_features': max_feats
            },
            {
                'clf': (GradientBoostingRegressor(
                    max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    random_state=0),),
                'clf__max_depth': [2,4],
                'clf__max_features': max_feats
            }
            ]
            if len(max_feats) <= 2:
                parameters += [{
                    'clf': [kernel_model],
                    'clf__linear__alpha': [100,10,1,0.1,0.001,0.0001]
                }]
            model = GridSearchCV(
                pipeline,
                parameters,
                scoring='neg_mean_squared_error',
                cv=CV,
                n_jobs=1,
                verbose=2)
        else:
            model = RandomForestRegressor(max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'], n_jobs=get_n_jobs(), random_state=0)
            # model = GradientBoostingRegressor(max_depth=4, n_estimators=200)
    print(f"outcome model (binary {is_binary})", model)
    return model
