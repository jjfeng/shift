"""
Code for comparators
"""

import logging
from typing import List, Dict
import tempfile
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from fsd_comparator import *
from scipy.stats import ks_2samp
from econml.grf import CausalForest
from causallearn.utils.cit import CIT
import pingouin

from sklearn.model_selection import train_test_split

from common import convert_prob_to_logit, compute_risk
from common import PipelineWeightedFit, get_n_jobs
from common import test_mmd
from decomp_explainer import ExplainerInference, InferenceResult
from estimate_datashifter import DetectorTestExplainer
from data_loader import DataLoader
from data_generator import DataGenerator

MAX_N_KCI = 3000  # Take N rows at random in KCI since it takes time to run
MAX_N_MMD = 3000

class ParametricChangeExplanation(ExplainerInference):
    """
    Fit a logistic regression model to explain outcomes from features in the pooled source and target data
    Return the importance of each feature as the coefficient of the interaction term between 
    dummy variable for the domain and the feature.
    Does not support feature subsets.
    """
    def __init__(self, ml_mdl, source_data_loader: DataLoader, target_data_loader: DataLoader, source_generator: DataGenerator, target_generator: DataGenerator, loss_func, combos, significance_level):
        self.ml_mdl = ml_mdl
        self.source_data_loader = source_data_loader
        self.source_generator = source_generator
        self.sourceX = self.source_data_loader._get_X()
        self.sourceY = self.source_data_loader._get_Y()
        self.target_data_loader = target_data_loader
        self.target_generator = target_generator
        self.targetX = self.target_data_loader._get_X()
        self.targetY = self.target_data_loader._get_Y()
        self.loss_func = loss_func
        self.combos = combos
        self.significance_level = significance_level
        self.explanation = []

        self.num_p = source_data_loader.num_p
        self.total_p = source_data_loader.total_p

    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceX, self.sourceY)
        source_dummy = np.zeros((self.sourceX.shape[0], 1))
        target_loss = self.loss_func(self.targetX, self.targetY)
        target_dummy = np.ones((self.targetX.shape[0], 1))

        lr_feats = np.concatenate([
            np.concatenate([source_dummy, self.sourceX, source_dummy * self.sourceX], axis=1),
            np.concatenate([target_dummy, self.targetX, target_dummy * self.targetX], axis=1),
        ])
        lr = pingouin.logistic_regression(
            lr_feats,
            np.concatenate([self.sourceY, self.targetY]),
            alpha=self.significance_level
        )
        print("LR CHANGE EXPLAIN", lr.coef)
        logging.info("parametric mechanism LR coeff %s pval %s", lr.coef, lr.pval)
        self.explanation_pvalue = lr.pval.to_list()[-self.sourceX.shape[1]:]  # pval for coeff of interaction terms
        self.explanation_statistic = lr.coef.to_list()[-self.sourceX.shape[1]:]

    def max_res_group(self, explanation):
        """
        Max of all feature explanations per group
        """
        explanation_group = -np.inf * np.ones(self.num_p, dtype=float)
        for feature, coef in enumerate(explanation):
            group = self.target_data_loader.vdict[feature]
            explanation_group[group] = max(coef, explanation_group[group])
        return explanation_group

    def get_detailed_res(self) -> pd.DataFrame:
        if self.target_data_loader.vdict is not None:
            self.explanation_pvalue = self.max_res_group(self.explanation_pvalue)
            self.explanation_statistic = self.max_res_group(self.explanation_statistic)

        df = []
        for i, value in enumerate(self.explanation_statistic):
            df.append(
                {
                    "value": value,
                    "level": "detail",
                    "decomp": "Cond_Outcome",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                    "pvalue": self.explanation_pvalue[i],
                }
            )
        return pd.DataFrame(df)

    def summary(self) -> pd.DataFrame:
        return self.get_detailed_res()
    
class ScoreMethod(ParametricChangeExplanation):
    """
    Feature shift detection method from Kulinski et al. 2020.
    Does not support feature subsets.
    """
    def do_decomposition(self):
        n_expectation = 30
        n_bootstrap_runs = 250
        alpha = 0.05

        print("SOURCE TARGET SHAPE", self.sourceX.shape, self.targetX.shape)
        self.sourceX_train, self.sourceX_test = train_test_split(
                self.sourceX, test_size=0.5, random_state=0
                )
        self.targetX_train, self.targetX_test = train_test_split(
                self.targetX, test_size=0.5, random_state=0
                )
        model = GaussianDensity()
        statistic = FisherDivergence(model, n_expectation=n_expectation)
        fsd = FeatureShiftDetectorPvalue(statistic, bootstrap_method='simple',
                           n_bootstrap_samples=n_bootstrap_runs,
                           significance_level=alpha)
        fsd.fit(self.sourceX_train, self.targetX_train)
        rng = np.random.RandomState(0)
        pvalues, _, _, scores = fsd.detect_and_localize(self.sourceX_test, self.targetX_test, random_state=rng, return_scores=True)
        self.explanation_statistic = scores
        self.explanation_pvalue = pvalues
        logging.info("ScoreMethod test statistic %s pvalue %s", self.explanation_statistic, self.explanation_pvalue)

    def get_detailed_res(self) -> pd.DataFrame:
        if self.target_data_loader.vdict is not None:
            self.explanation_pvalue = self.max_res_group(self.explanation_pvalue)
            self.explanation_statistic = self.max_res_group(self.explanation_statistic)

        df = []
        for i, test_statistic in enumerate(self.explanation_statistic):
            df.append(
                {
                    "value": test_statistic,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                    "pvalue": self.explanation_pvalue[i],
                }
            )
        return pd.DataFrame(df)
    
class DomainClassifierExplanation(ParametricChangeExplanation):
    def do_decomposition(self):
        NUM_ESTIMATORS = 200
        CV = 3
        pipeline = PipelineWeightedFit([
            ('clf', RandomForestClassifier())
        ])
        parameters = [
            {
                'clf': (RandomForestClassifier(
                    max_depth=6,
                    criterion='log_loss',
                    max_features=None,
                    n_estimators=NUM_ESTIMATORS,
                    n_jobs=get_n_jobs(),
                    random_state=0),),
                'clf__max_depth': [2,4,6,8],
                'clf__max_features': ['sqrt',None]
            }
        ]
        domain_classifier = GridSearchCV(pipeline, parameters, n_jobs=1, cv=CV, scoring="neg_log_loss", verbose=2)
        domain_classifier.fit(
            np.concatenate([self.sourceX, self.targetX]),
            np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        )
        self.explanation = domain_classifier.best_estimator_.named_steps['clf'].feature_importances_
        logging.info("DomainClassifierExplanation %s", self.explanation)

    def get_detailed_res(self) -> pd.DataFrame:
        if self.target_data_loader.vdict is not None:
            self.explanation = self.max_res_group(self.explanation)

        df = []
        for i, coef in enumerate(self.explanation):
            df.append(
                {
                    "value": coef,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                    "pvalue": "",
                }
            )
        return pd.DataFrame(df)

def random_sample_array(X, n):
    """
    Randomly sample n rows from X
    """
    if X.shape[0] <= n:
        return X
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle[:n]]

class MMDCovariateTestAgg(ParametricChangeExplanation):
    def do_decomposition(self):
        X_tr = self.sourceX
        X_te = self.targetX

        X_tr = random_sample_array(X_tr, MAX_N_MMD)
        X_te = random_sample_array(X_te, MAX_N_MMD)

        # MMD
        decision, pvalue, mmd = test_mmd(X_tr, X_te, alpha=self.significance_level)
        self.explanation_decision = decision
        self.explanation_pvalue = pvalue
        self.explanation_statistic = mmd
        logging.info("MMDCovariateTest statistic %s decision %s pvalue %s", self.explanation_statistic, self.explanation_decision, self.explanation_pvalue)

    def get_detailed_res(self) -> pd.DataFrame:
        df = [{
                "value": self.explanation_statistic,
                "level": "agg",
                "decomp": "Covariate",
                "vars": "X",
                "est": type(self).__name__,
                "pvalue": self.explanation_pvalue,
                "decision": self.explanation_decision,
        }]
        return pd.DataFrame(df)
    
class KCICovariateTestAgg(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
        ]
        if data.shape[0] > MAX_N_KCI:
            shuffle = np.random.permutation(data.shape[0])
            data = data[shuffle[:MAX_N_KCI],:]
        kci_obj = CIT(data, "kci")
        pvalue = kci_obj(data.shape[1]-1, np.arange(all_sourceX_targetX.shape[1]), [])
        self.explanation = pvalue
        logging.info("KCICovariateTest %s", self.explanation)
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = [{
                "value": self.explanation,
                "level": "agg",
                "decomp": "Covariate",
                "vars": "X",
                "est": type(self).__name__,
                "pvalue": self.explanation,
        }]
        return pd.DataFrame(df)
        
class KCICovariateTest(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
        ]
        if data.shape[0] > MAX_N_KCI:
            shuffle = np.random.permutation(data.shape[0])
            data = data[shuffle[:MAX_N_KCI],:]
        kci_obj = CIT(data, "kci")
        X_indices = np.arange(all_sourceX_targetX.shape[1])
        pvalues = []
        vars_names = []
        if self.combos is not None:
            assert len(self.combos[0])==all_sourceX_targetX.shape[1], "length of combos should be equal to the number of features"
            for subgroup_mask in self.combos:
                Xs_indices = [j for j in X_indices if subgroup_mask[j]]
                Xms_indices = [j for j in X_indices if j not in Xs_indices]
                logging.info("KCICovariateTest subgroup %s given %s", Xs_indices, Xms_indices)
                pvalue = kci_obj(data.shape[1]-1, Xs_indices, Xms_indices)
                pvalues.append(pvalue)
                vars_names.append(",".join(["X%d" % (i+1) for i in Xs_indices]))
        else:  # test each feature individually if combos not given
            for i in X_indices:
                Xms_indices = [j for j in X_indices if j!=i]
                pvalue = kci_obj(data.shape[1]-1, i, Xms_indices)
                pvalues.append(pvalue)
                vars_names.append("X%d" % (i+1))
        self.explanation_pvalue = pvalues
        self.vars_names = vars_names
        logging.info("KCICovariateTest %s", self.explanation_pvalue)
        self.explanation_statistic = np.ones_like(self.explanation_pvalue)*np.nan  # no test statistic
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, value in enumerate(self.explanation_statistic):
            df.append(
                {
                    "value": value,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": self.vars_names[i],
                    "est": type(self).__name__,
                    "pvalue": self.explanation_pvalue[i],
                }
            )
        return pd.DataFrame(df)
    
class MMDOutcomeTestAgg(ParametricChangeExplanation, DetectorTestExplainer):
    def do_decomposition(self):
        # Covaiates, loss matrices
        source_loss = self.loss_func(self.sourceX, self.sourceY)
        target_loss = self.loss_func(self.targetX, self.targetY)
        X_tr = np.concatenate([
            self.sourceX,
            source_loss.reshape(-1,1),
        ], axis=1)
        X_te = np.concatenate([
            self.targetX,
            target_loss.reshape(-1,1),
        ], axis=1)

        X_tr = random_sample_array(X_tr, MAX_N_MMD)
        X_te = random_sample_array(X_te, MAX_N_MMD)
        
        decision, pvalue, mmd = test_mmd(X_tr, X_te, alpha=self.significance_level)
        self.explanation_decision = decision
        self.explanation_pvalue = pvalue
        self.explanation_statistic = mmd
        logging.info("MMDOutcomeTest statistic %s decision %s pvalue %s", self.explanation_statistic, self.explanation_decision, self.explanation_pvalue)
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = [{
                "value": self.explanation_statistic,
                "level": "agg",
                "decomp": "Outcome",
                "vars": "Y",
                "est": type(self).__name__,
                "pvalue": self.explanation_pvalue,
                "decision": self.explanation_decision,
        }]
        return pd.DataFrame(df)
    
class KCIOutcomeTestAgg(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        all_sourceY_targetY = np.concatenate([self.sourceY, self.targetY])
        all_source_target_loss = self.loss_func(all_sourceX_targetX, all_sourceY_targetY)
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
            all_source_target_loss
        ]
        if data.shape[0] > MAX_N_KCI:
            shuffle = np.random.permutation(data.shape[0])
            data = data[shuffle[:MAX_N_KCI],:]
        kci_obj = CIT(data, "kci")
        pvalue = kci_obj(data.shape[1]-2, data.shape[1]-1, np.arange(all_sourceX_targetX.shape[1]))
        self.explanation = pvalue
        logging.info("KCIOutcomeTest %s", self.explanation)
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = [{
                "value": self.explanation,
                "level": "agg",
                "decomp": "Outcome",
                "vars": "Y",
                "est": type(self).__name__,
                "pvalue": self.explanation,
        }]
        return pd.DataFrame(df)
    
class KCIOutcomeTest(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        all_sourceY_targetY = np.concatenate([self.sourceY, self.targetY])
        all_source_target_loss = self.loss_func(all_sourceX_targetX, all_sourceY_targetY)
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
            all_source_target_loss
        ]
        if data.shape[0] > MAX_N_KCI:
            shuffle = np.random.permutation(data.shape[0])
            data = data[shuffle[:MAX_N_KCI],:]
        kci_obj = CIT(data, "kci")
        X_indices = np.arange(all_sourceX_targetX.shape[1])
        pvalues = []
        vars_names = []
        if self.combos is not None:
            assert len(self.combos[0])==all_sourceX_targetX.shape[1], "length of combos should be equal to the number of features"
            for subgroup_mask in self.combos:
                Xs_indices = [j for j in X_indices if subgroup_mask[j]]
                Xms_indices = [j for j in X_indices if j not in Xs_indices]
                logging.info("KCIOutcomeTest subgroup %s given %s", Xs_indices, Xms_indices)
                pvalue = kci_obj(data.shape[1]-2, data.shape[1]-1, Xms_indices)
                pvalues.append(pvalue)
                vars_names.append(",".join(["X%d" % (i+1) for i in Xs_indices]))
        else:  # test each feature individually if combos not given
            for i in X_indices:
                Xms_indices = [j for j in X_indices if j!=i]
                pvalue = kci_obj(data.shape[1]-2, data.shape[1]-1, Xms_indices)
                pvalues.append(pvalue)
                vars_names.append("X%d" % (i+1))
        self.explanation_pvalue = pvalues
        self.vars_names = vars_names
        logging.info("KCIOutcomeTest %s", self.explanation_pvalue)
        self.explanation_statistic = np.ones_like(self.explanation_pvalue)*np.nan
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, value in enumerate(self.explanation_statistic):
            df.append(
                {
                    "value": value,
                    "level": "detail",
                    "decomp": "Cond_Outcome",
                    "vars": self.vars_names[i],
                    "est": type(self).__name__,
                    "pvalue": self.explanation_pvalue[i],
                }
            )
        return pd.DataFrame(df)
    
class LinearMediationTest(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        all_sourceY_targetY = np.concatenate([self.sourceY, self.targetY])
        all_source_target_loss = self.loss_func(all_sourceX_targetX, all_sourceY_targetY)
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
            all_source_target_loss
        ]
        data = pd.DataFrame(data, columns=np.arange(data.shape[1]).astype(str))
        tcol = data.columns[-2]
        ycol = data.columns[-1]
        allconfcols = data.columns[:-2].tolist()  # all confounders

        data[ycol] = data[ycol].astype(bool)

        results = []
        for c in np.arange(all_sourceX_targetX.shape[1]):
            mcol = data.columns[c:(c+1)].tolist()
            confcols = [i for i in allconfcols if i not in mcol]  # all confounders except mediators
            result = pingouin.mediation_analysis(
                data=data,
                x=tcol,
                m=mcol,
                y=ycol,
                covar=confcols,
                alpha=self.significance_level,
                seed=0
            )
            assert result.iloc[-1]['path']=='Indirect', 'incorrect output of mediation analysis'
            results.append((result.iloc[-1]['coef'], result.iloc[-1]['pval']))
        self.explanation_statistic, self.explanation_pvalue = zip(*results)
        logging.info("LinearMediationTest coef %s pvalue %s", self.explanation_statistic, self.explanation_pvalue)

    def get_detailed_res(self) -> pd.DataFrame:
        if self.target_data_loader.vdict is not None:
            self.explanation_statistic = self.max_res_group(self.explanation_statistic)
            self.explanation_pvalue = self.max_res_group(self.explanation_pvalue)  # TODO: return mediation by variable group

        df = []
        for i, test_statistic in enumerate(self.explanation_statistic):
            df.append({
                "value": test_statistic,
                "level": "detail",
                "decomp": "Cond_Cov",
                "vars": "X%d" % (i+1),
                "est": type(self).__name__,
                "pvalue": self.explanation_pvalue[i],
            })
        return pd.DataFrame(df)
        
class TEVIMTest(ParametricChangeExplanation):
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        all_sourceY_targetY = np.concatenate([self.sourceY, self.targetY])
        all_source_target_loss = self.loss_func(all_sourceX_targetX, all_sourceY_targetY)
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])
        data = np.c_[
            all_sourceX_targetX,
            domain_indicator,
            all_source_target_loss
        ]
        cols = np.arange(1, data.shape[1]+1).astype(str)  # 1-indexed columns for R
        covariate_cols = cols[:-2]
        data = pd.DataFrame(data, columns=cols)
        if self.combos is not None:
            assert len(self.combos[0])==all_sourceX_targetX.shape[1], "length of combos should be equal to the number of features"
            subgroup_masks = [",".join(covariate_cols[mask]) for mask in self.combos]  # combos are the variables to exclude
            var_names = [",".join(["X"+i for i in covariate_cols[mask]]) for mask in self.combos]
        else:
            subgroup_masks = covariate_cols.tolist()
            var_names = ["X%d" % (i+1) for i in range(all_sourceX_targetX.shape[1])]

        # Create files
        with tempfile.NamedTemporaryFile() as fr:
            data.to_csv(fr, index=False)
            fr.seek(0)
            print(pd.read_csv(fr.name))
            with tempfile.NamedTemporaryFile() as fw:
                cmd = [
                    'Rscript',
                    '--vanilla',
                    'tevims/tevim.R',
                    fr.name,
                    fw.name,
                    "+".join(subgroup_masks),
                    "TRUE"
                ]
                cmd = list(map(str, cmd))
                print("Calling:", " ".join(cmd))
                res = subprocess.call(cmd)
                assert res==0, "Rscript did not run"
                fw.seek(0)
                result = pd.read_csv(fw.name)

        print(result)
        logging.info("TEVIMTest coef, pval %s", result)
        result = result[
            # (result["algorithm"]=="0D") &
            (result["algorithm"]=="CF") &
            (result["scale"]=="u")
        ]
        print(result)
        self.explanation_statistic, self.explanation_pvalue = zip(*result[["tevim","pval"]].values)
        self.vars_names = var_names
        logging.info("TEVIMTest coef %s pvalue %s", self.explanation_statistic, self.explanation_pvalue)
    
    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, test_statistic in enumerate(self.explanation_statistic):
            df.append({
                "value": test_statistic,
                "level": "detail",
                "decomp": "Cond_Outcome",
                "vars": self.vars_names[i],
                "est": type(self).__name__,
                "pvalue": self.explanation_pvalue[i],
            })
        return pd.DataFrame(df)
    
class KSTest(ParametricChangeExplanation):
    def do_decomposition(self):
        test_stats = ks_2samp(self.sourceX, self.targetX, axis=0)
        self.explanation = {
            "statistic": test_stats.statistic,
            "pvalue": test_stats.pvalue,
        }
        logging.info("KSTest %s", self.explanation["pvalue"])
    
    def get_detailed_res(self) -> pd.DataFrame:
        if self.target_data_loader.vdict is not None:
            self.explanation_statistic = self.max_res_group(self.explanation["statistic"])
            self.explanation_pvalue = self.max_res_group(self.explanation["pvalue"])

        df = []
        for i, test_statistic in enumerate(self.explanation_statistic):
            df.append(
                {
                    "value": test_statistic,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                    "pvalue": self.explanation_pvalue[i],
                }
            )
        return pd.DataFrame(df)

class ParametricAccExplanation(ParametricChangeExplanation):
    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceX, self.sourceY)
        source_dummy = np.zeros((self.sourceX.shape[0], 1))
        target_loss = self.loss_func(self.targetX, self.targetY)
        target_dummy = np.ones((self.targetX.shape[0], 1))
        lr_feats = np.concatenate([
            np.concatenate([source_dummy, self.sourceX, source_dummy * self.sourceX], axis=1),
            np.concatenate([target_dummy, self.targetX, target_dummy * self.targetX], axis=1),
        ])

        try:
            if set(np.unique(source_loss.astype(int)))==set([0,1]):
                logging.info("Binary outcome. Calling logistic regression")
                acc_lr = pingouin.logistic_regression(
                    lr_feats,
                    np.concatenate([source_loss, target_loss]),
                    alpha=self.significance_level
                )
            else:
                logging.info("Outcome not binary. Calling linear regression")
                acc_lr = pingouin.linear_regression(
                    lr_feats,
                    np.concatenate([source_loss, target_loss]),
                    alpha=self.significance_level
                )
        except:
            raise NotImplementedError
        print("LR LOSS EXPLAIN", acc_lr.coef)
        logging.info("parametric LR loss explain coef %s pvalue %s", acc_lr.coef, acc_lr.pval)
        self.explanation_statistic = acc_lr.coef.to_list()[-self.sourceX.shape[1]:]
        self.explanation_pvalue = acc_lr.pval.to_list()[-self.sourceX.shape[1]:]

class RandomForestExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        rf_source = RandomForestClassifier()
        rf_source.fit(self.sourceX, self.sourceY)
        rf_target = RandomForestClassifier()
        rf_target.fit(self.targetX, self.targetY)
        
        source_vi = rf_source.feature_importances_
        source_vi = source_vi/source_vi.sum()
        target_vi = rf_target.feature_importances_
        target_vi = target_vi/target_vi.sum()

        self.explanation_statistic = np.abs(target_vi - source_vi)
        self.explanation_statistic /= self.explanation_statistic.sum()
        print("RF EXPLAIN", self.explanation_statistic)
        self.explanation_pvalue = np.ones_like(self.explanation_statistic)*np.nan
        logging.info("RF explain %s", self.explanation_statistic)
        
class CausalForestExplanation(ParametricAccExplanation):
    """
    Feature importance of a causal forest fitted to estimate effect of domain on loss
    """
    def do_decomposition(self):
        all_sourceX_targetX = np.concatenate([self.sourceX, self.targetX])
        all_sourceY_targetY = np.concatenate([self.sourceY, self.targetY])
        all_source_target_loss = self.loss_func(all_sourceX_targetX, all_sourceY_targetY)
        domain_indicator = np.concatenate([np.zeros(self.sourceX.shape[0]), np.ones(self.targetX.shape[0])])

        grf = CausalForest(
            n_estimators=100,
            max_depth=None,
        )
        grf.fit(
            X=all_sourceX_targetX,
            T=domain_indicator,
            y=all_source_target_loss
        )
        self.explanation_statistic = grf.feature_importances_
        self.explanation_statistic /= self.explanation_statistic.sum()
        print("CausalForest EXPLAIN", self.explanation_statistic)
        logging.info("CausalForest explain %s", self.explanation_statistic)
        self.explanation_pvalue = np.ones_like(self.explanation_statistic)*np.nan

class GBTAccExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        target_loss = self.loss_func(self.targetX, self.targetY)
        pred_logit_target = convert_prob_to_logit(self.ml_mdl.predict_proba(self.targetX)[:,1:])

        acc_mdl = GridSearchCV(
            GradientBoostingClassifier(init=LogisticRegression(penalty="l1", solver="saga", max_iter=20000), n_estimators=50, max_depth=2),
            param_grid={
                'n_estimators': [50,100,200],
            },
            n_jobs=-1
        )
        feats = np.concatenate([pred_logit_target, self.targetX], axis=1)
        acc_mdl.fit(feats, target_loss)
        self.explanation_statistic = acc_mdl.best_estimator_.feature_importances_[-self.total_p:]
        print("GBT EXPLAIN", self.explanation_statistic)
        logging.info("GBT explain %s", self.explanation)
        logging.info("GBT explain %s", acc_mdl.best_estimator_.feature_importances_)
        self.explanation_pvalue = np.ones_like(self.explanation_statistic)*np.nan

class RandomForestAccExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceX, self.sourceY)
        source_dummy = np.zeros((self.sourceX.shape[0], 1))
        target_loss = self.loss_func(self.targetX, self.targetY)
        target_dummy = np.ones((self.targetX.shape[0], 1))
        pred_logit_source = convert_prob_to_logit(self.ml_mdl.predict_proba(self.sourceX)[:,1:])
        pred_logit_target = convert_prob_to_logit(self.ml_mdl.predict_proba(self.targetX)[:,1:])

        acc_mdl = RandomForestRegressor()
        feats = np.concatenate([
            np.concatenate([pred_logit_source, source_dummy, self.sourceX], axis=1),
            np.concatenate([pred_logit_target, target_dummy, self.targetX], axis=1)
        ], axis=0)
        acc_mdl.fit(
            feats,
            np.concatenate([source_loss, target_loss]))
        self.explanation_statistic = acc_mdl.feature_importances_[-self.total_p:]
        print("RF ACC EXPLAIN", self.explanation_statistic)
        logging.info("RF ACC explain %s", self.explanation_statistic)
        logging.info("RF ACC explain %s", acc_mdl.feature_importances_)
        self.explanation_pvalue = np.ones_like(self.explanation_statistic)*np.nan