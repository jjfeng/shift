"""Computes shift explanations by models estimated from data.
Does not use data generators.
"""

import time
import logging
from typing import Tuple, Dict
from tqdm import tqdm
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
import sklearn

from decomp_explainer import BaseShiftExplainer, InferenceResult, to_str_inf_res
from data_loader import DataLoader
from data_loader import train_test_loader_split
from detectors import DetectorCovariateShift
from common import *


OUTCOME_MODEL_ARGS = {
    'n_estimators': 400, 
    'max_features': 'sqrt', 
    'max_depth': 16, 
    'bootstrap': True
}
DETECTORS = [
    # LinearRegression(),
    # Ridge(alpha=1, max_iter=20000),
    # RandomForestRegressor(max_depth=2, n_estimators=400),
    # RandomForestRegressor(max_depth=4, n_estimators=400),
    # RandomForestRegressor(max_depth=6),
    GradientBoostingRegressor(max_depth=2, n_estimators=200),
]
MAX_DETECTORS_COV = 5


class DetectorTestExplainer(BaseShiftExplainer):
    # split_ratio = 0.8  # for aggregate comparisons
    split_ratio = 0.5
    candidate_omegas = np.linspace(0, 100, num=2000)
    candidate_lambdas = np.linspace(-5, 5, num=500)
    def __init__(self, source_loader: DataLoader, target_loader: DataLoader, source_data_generator, target_data_generator, loss_func, ml_mdl, do_grid_search, do_clipping, gridsearch_polynom_lr: bool=False, is_oracle: bool=False, tolerance: float=0.0, prevalence: float=0.0, num_bins: int=0, test_greater: bool=True, filter_independent_features: bool=False, filter_significance_level: float=0.05):
        # Data
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_data_generator = source_data_generator
        self.target_data_generator = target_data_generator
        # Loss function
        self.loss_func = loss_func
        self.ml_mdl = ml_mdl
        # Hypothesis test parameters
        self.do_grid_search = do_grid_search
        self.domain_prob_cutoff = 1e-3 if do_clipping else 1e-10
        self.ustat_prob_cutoff = 1e-3 if do_clipping else 1e-10
        self.gridsearch_polynom_lr = gridsearch_polynom_lr
        self.is_oracle = is_oracle
        self.tolerance = tolerance
        self.min_prevalence = prevalence
        self.num_bins = num_bins  # bins for expected outcome function
        self.test_greater = test_greater
        self.filter_independent_features = filter_independent_features
        self.filter_significance_level = filter_significance_level
        self.correlated_features = None
        # Models
        self.source_outcome_model = None
        self.target_outcome_model = None
        self.density_x_model = None
        self.source_outcome_filter_model = None
        self.density_x_filter_model = None
        self.binning = None

        # Data
        # Split data into train and test
        self.target_train, self.target_test = train_test_loader_split(
            self.target_loader, test_size=self.split_ratio
        )
        self.source_train, self.source_test = train_test_loader_split(
            self.source_loader, test_size=self.split_ratio
        )
        self.test_source_n = self.source_test.X.shape[0]
        self.test_target_n = self.target_test.X.shape[0]
        self.source_prevalence = self.test_source_n/(self.test_source_n + self.test_target_n)
        self.target_prevalence = 1 - self.source_prevalence
        self.domain_ratio_train = self.source_train.X.shape[0] / self.target_train.X.shape[0]
        logging.info("prevalence source %f, target %f", self.source_prevalence, self.target_prevalence)
        
        # Save detectors
        self.agg_detectors_x = None
        self.agg_detectors_y = None
        self.agg_correlated_features = None  # features in covariate test
        self.detail_detectors_x = dict()  # {'subgroup mask': (detector,omega), 'correlated_features': [True, True, ...]}
        self.detail_detectors_y = dict()  # {'subgroup mask': detector, 'correlated_features': [True, True, ...]}
        # Save results
        self.agg_res_x = None
        self.agg_res_y = None
        self.detail_covariate_res_dict = {}
        self.detail_cond_outcome_res_dict = {}

    @property
    def residual_sign(self):
        return 1 if self.test_greater else -1
    
    def _prepare_nuisance_models(self, filter_independent_features: bool=False):
        """Fit outcome and density models on all and filtered features
        """
        # Fit nuisance models
        if (self.source_outcome_model is None) or (self.target_outcome_model is None) or (self.density_x_model is None):
            self.source_outcome_model, self.target_outcome_model, self.density_x_model = self._estimate_nuisance_models(
                self.source_train,
                self.target_train
            )

        # Find features correlated with loss function for the composite covariate test
        if self.correlated_features is None and filter_independent_features:
            # If we have not computed correlated features
            self.correlated_features = self._compute_correlated_features(self.source_train)
        elif not filter_independent_features:
            # Keep all features
            self.correlated_features = np.ones(self.source_train.X.shape[1], dtype=bool)

        self.source_train_filter = self.source_train.copy()
        self.source_train_filter.subset_X(self.correlated_features)  # keeps mdl_X features constant
        self.target_train_filter = self.target_train.copy()
        self.target_train_filter.subset_X(self.correlated_features)
        self.source_test_filter = self.source_test.copy()
        self.source_test_filter.subset_X(self.correlated_features)
        self.target_test_filter = self.target_test.copy()
        self.target_test_filter.subset_X(self.correlated_features)

        # Refit nuisance models on correlated features
        if (self.source_outcome_filter_model is None) or (self.density_x_filter_model is None):
            # Re-fit models only if at least one feature is removed
            if not np.all(self.correlated_features):
                self.source_outcome_filter_model, _, self.density_x_filter_model = self._estimate_nuisance_models(
                    self.source_train_filter,
                    self.target_train_filter,
                )
            else:
                self.source_outcome_filter_model = self.source_outcome_model
                self.density_x_filter_model = self.density_x_model
        
        # Bin source outcome probability for the outcome tests
        if self.binning is None:
            self.binning = self._get_binning_fn(
                self.source_train_filter.X, 
                self.target_train_filter.X,
                self.source_outcome_filter_model if self.filter_independent_features else self.source_outcome_model, 
                self.num_bins
            )
    
    def _compute_correlated_features(self, source_train: DataLoader):
        """Get features correlated with the loss function for modified covariate test
        """
        # Remove features independent of loss function
        # TODO: use a marginal independence test
        correlated_features, pvalues = get_correlated_features(
            X=source_train.X, 
            Y=self.loss_func(source_train.mdl_X, source_train.Y), 
            alpha=self.filter_significance_level
        )
        logging.info("Correlated features %s", correlated_features)
        logging.info("pvalues for independence test %s", pvalues)
        return correlated_features

    def _estimate_nuisance_models(self, source_train: DataLoader, target_train: DataLoader):
        # Fit outcome models for Y|X in source and target domains
        source_outcome_model, target_outcome_model = self._estimate_outcome_models(
            source_train, target_train
        )
    
        # Fit density ratio models for X
        density_x_model = self._estimate_density_model(
            source_train.X, target_train.X
        )

        return source_outcome_model, target_outcome_model, density_x_model
    
    def _get_binning_fn(self, sourceX_train: np.ndarray, targetX_train: np.ndarray, source_outcome_model, num_bins: int):
        """Bin logits of source outcome probability
        """
        # Set max value of expected outcome in source to decide range for binning if not [0,1]
        all_sourceX_targetX_train = np.concatenate([sourceX_train, targetX_train])
        source_prob = self._get_probability(source_outcome_model, all_sourceX_targetX_train)
        # exp_loss_source = compute_risk(self.loss_func, self.all_sourceX_targetX_train, source_prob)  # give mdl_X
        binning = Binning(
            init_prob=convert_prob_to_logit(source_prob),
            num_bins=num_bins, 
            type='space'
        )
        logging.info("BIN CENTERS for exp loss in source %s", binning.centers)
        return binning

    def _estimate_density_model(self, sourceX_train: np.ndarray, targetX_train: np.ndarray):
        """Fit density models for X
        """
        density_x_model = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(targetX_train.shape[1]),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        density_x_model.fit(
            np.concatenate([sourceX_train, targetX_train]),
            np.concatenate([np.zeros(sourceX_train.shape[0]), np.ones(targetX_train.shape[0])])
        )
        if self.do_grid_search:
            logging.info("selected model densityX %s %s", density_x_model.best_estimator_, density_x_model.cv_results_)
        
        # self.density_model_x = lambda x: self.target_data_generator._get_density_X(x)/self.source_data_generator._get_density_X(x)
        # self.density_model_xs = lambda x, subgroup_mask: self.target_data_generator._get_density_Xs(x, subgroup_mask)/self.source_data_generator._get_density_Xs(x, subgroup_mask)

        return density_x_model

    def _estimate_outcome_models(self, source_train: DataLoader, target_train: DataLoader):
        """Fit target and source outcome models
        """
        # Fit target outcome model
        target_outcome_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(target_train.X.shape[1]),
            model_args=OUTCOME_MODEL_ARGS,
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        target_outcome_model.fit(
            target_train.X, target_train.Y
        )

        # Fit source outcome model
        source_outcome_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(source_train.X.shape[1]),
            model_args=OUTCOME_MODEL_ARGS,
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        source_outcome_model.fit(
            source_train.X, source_train.Y
        )
        if self.do_grid_search and not self.is_oracle:
            logging.info("CV RESULTS %s", source_outcome_model.cv_results_)
            print("selected model sourceY", source_outcome_model.best_estimator_, source_outcome_model.best_score_)
            logging.info("selected model sourceY %s %f", source_outcome_model.best_estimator_, source_outcome_model.best_score_)
            logging.info("CV RESULTS %s", target_outcome_model.cv_results_)
            print("selected model targetY", target_outcome_model.best_estimator_, target_outcome_model.best_score_)
            logging.info("selected model targetY %s %f", target_outcome_model.best_estimator_, target_outcome_model.best_score_)

        return source_outcome_model, target_outcome_model

    def _estimate_density_model_xs(self, sourceX_train: np.ndarray, targetX_train: np.ndarray, subgroup_mask: np.ndarray, detections_source_target: np.ndarray=None):
        """Fits density ratio model for X_s
        """
        feats = np.concatenate([
            sourceX_train[:, subgroup_mask], targetX_train[:, subgroup_mask]
        ], axis=0)
        labels = np.concatenate([
            np.zeros(sourceX_train.shape[0]), np.ones(targetX_train.shape[0])
        ], axis=0)
        if detections_source_target is not None:
            feats = np.concatenate([
                feats, detections_source_target[:, np.newaxis]
            ], axis=1)
            
        density_model_xs = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(feats.shape[1]),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        density_model_xs.fit(feats, labels)
        if self.do_grid_search:
            logging.info("selected model density_model_xs %s %s", density_model_xs.best_estimator_, density_model_xs.cv_results_)
        return density_model_xs
    
    def _test_covariate_aggregate(self, source_train: DataLoader, source_test: DataLoader, target_train: DataLoader, target_test: DataLoader, source_outcome_model, density_x_model) -> Dict[str, InferenceResult]:
        """
        Test for loss difference in a sufficiently large subgroup due to covariate shift
        """
        ## FIT models
        # Cache exp loss and odds to reuse in predict()
        cached_exp_loss_odds_source_train = self.get_exp_loss_odds_detector(source_train, source_outcome_model, density_x_model, None, None)
        cached_exp_loss_odds_target_train = self.get_exp_loss_odds_detector(target_train, source_outcome_model, density_x_model, None, None)
        detectors_omega = self._estimate_detectors_covariate_aggregate(source_train, target_train, source_outcome_model, density_x_model, cached_exp_loss_odds_source_train, cached_exp_loss_odds_target_train)  # returns list of (detector, omega)
        logging.info("detectors above min prevalence %s", detectors_omega)
        
        ## EVALUATE plugin
        # On source
        source_loss = self.loss_func(source_test.mdl_X, source_test.Y)
        exp_loss_source_on_sourceX = self._compute_risk(source_outcome_model, source_test.X, source_test.mdl_X)  # Z_0(x) for x in source
        odds_on_sourceX = self._get_density_ratio_x(density_x_model, source_test.X)
        cached_exp_loss_odds_source_test = self.get_exp_loss_odds_detector(source_test, source_outcome_model, density_x_model, None, None)

        # On target
        exp_loss_source_on_targetX = self._compute_risk(source_outcome_model, target_test.X, target_test.mdl_X)  # Z_0(x) for x in target
        cached_exp_loss_odds_target_test = self.get_exp_loss_odds_detector(target_test, source_outcome_model, density_x_model, None, None)
        
        all_plugins = []
        all_ifs = []
        all_detection_masks = []
        if len(detectors_omega) == 0:
            all_plugins.append(-np.ones(self.test_source_n+self.test_target_n))
            all_ifs.append(-np.ones(self.test_source_n+self.test_target_n))
            all_detection_masks.append(np.zeros(self.test_source_n + self.test_target_n))
            logging.info("COV detections below threshold %s", self.min_prevalence)
        else:
            for (detector, omega) in detectors_omega:
                logging.info("detector omega %s lambda %s", omega, detector.min_lambda)
                detection_source = detector.predict(source_test, omega, cached_exp_loss_odds_source_test) 
                source_mask = detection_source > 0
                detection_target = detector.predict(target_test, omega, cached_exp_loss_odds_target_test)
                logging.info("detections source %s target %s", detection_source, detection_target)
                target_mask = detection_target > 0
                logging.info("omega actual %s estimated %s",omega, np.mean(source_mask)/np.mean(target_mask))
                detection_prevalence_target = sum(target_mask) / (sum(target_mask) + sum(source_mask))
                detection_prevalence_source = 1 - detection_prevalence_target
                all_plugins.append(
                    np.concatenate([
                        (-self.residual_sign * source_loss - self.tolerance) / detection_prevalence_source,
                        (self.residual_sign * exp_loss_source_on_targetX) / detection_prevalence_target
                    ])
                )
                all_ifs.append(
                    np.concatenate([
                        (-self.residual_sign * (source_loss - (source_loss - exp_loss_source_on_sourceX) * odds_on_sourceX) - self.tolerance) / detection_prevalence_source,
                        (self.residual_sign * exp_loss_source_on_targetX) / detection_prevalence_target
                    ])
                )
                all_detection_masks.append(np.concatenate([source_mask, target_mask]))
                logging.info("AGGREGATE COV detection proportions source %s target %s", np.mean(source_mask), np.mean(target_mask))
                logging.info("AGGREGATE COV exp loss source %s target %s", np.mean(source_loss[source_mask]), np.mean(exp_loss_source_on_targetX[target_mask]))
                logging.info("AGGREGATE COV plugin %s correction term %s", all_plugins[-1][all_detection_masks[-1].astype(bool)].mean(), ((source_loss - exp_loss_source_on_sourceX) * odds_on_sourceX)[source_mask].mean())

        self.agg_detectors_x = detectors_omega
        self.agg_correlated_features = self.correlated_features

        # TODO: get features with non zero coefficients in source outcome and density models
        features_in_detectors = np.ones(source_test.X.shape[1], dtype=bool)  # assuming all features in correlated features mask

        # Get pvalues from testing whether features used in detector are independent of loss function
        if self.filter_independent_features:
            _, pvalues_correlated_features = get_correlated_features(
                X=source_test.X[:, features_in_detectors], 
                Y=self.loss_func(source_test.mdl_X, source_test.Y), 
                alpha=self.filter_significance_level
            )
        else:
            pvalues_correlated_features = None
        logging.info("pvalues for features in detectors %s", pvalues_correlated_features)

        return {
            'plugin': InferenceResult(
                all_plugins,
                all_detection_masks,
                pvalues_correlated_features
            ),
            'onestep': InferenceResult(
                all_ifs,
                all_detection_masks,
                pvalues_correlated_features
            )
        }

    def _test_cond_outcome_aggregate(self, target_train: DataLoader, source_test: DataLoader, target_test: DataLoader, source_outcome_model, density_x_model):
        """
        Test for a loss difference in a sufficiently large subgroup due to conditional outcome shift
        """
        # FIT models
        detectors = self._estimate_detectors_cond_outcome_aggregate(target_train)

        # EVALUATE
        # On target
        target_loss = self.loss_func(target_test.mdl_X, target_test.Y)
        source_prob_on_targetX = self._get_probability(source_outcome_model, target_test.X)
        exp_loss_source_on_targetX = compute_risk(self.loss_func, target_test.mdl_X, source_prob_on_targetX)
        residual_remaining = self.residual_sign * (target_loss - exp_loss_source_on_targetX)

        # On source
        source_loss = self.loss_func(source_test.mdl_X, source_test.Y)
        source_prob_on_sourceX = self._get_probability(source_outcome_model, source_test.X)
        exp_loss_source_on_sourceX = compute_risk(self.loss_func, source_test.mdl_X, source_prob_on_sourceX)
        odds_target_x = self._get_density_ratio_x(density_x_model, source_test.X)
        logging.info("COND OUTCOME ODDs X min %f max %f", odds_target_x.min(), odds_target_x.max())

        all_plugins = []
        all_ifs = []
        all_detection_masks_plugin = []
        all_detection_masks_onestep = []
        for detector in detectors:
            detection_on_targetX = detector.predict(target_test.X)
            detected_residual_on_targetX = (residual_remaining - self.tolerance) * (detection_on_targetX > 0)
            
            detection_on_sourceX = detector.predict(source_test.X)
            correction_term_source = -1 * self.residual_sign * (source_loss - exp_loss_source_on_sourceX) * odds_target_x * (detection_on_sourceX > 0)
            
            detection_prevalence = np.mean(detection_on_targetX > 0)
            if detection_prevalence <= self.min_prevalence:
                all_plugins.append(-np.ones(self.test_source_n+self.test_target_n))
                all_ifs.append(-np.ones(self.test_source_n+self.test_target_n))
                all_detection_masks_plugin.append(np.zeros(self.test_source_n + self.test_target_n))
                all_detection_masks_onestep.append(np.zeros(self.test_source_n + self.test_target_n))
                logging.info("COND OUTCOME detections below threshold %s: %s", self.min_prevalence, detection_prevalence)
            else:
                target_mask = detection_on_targetX > 0
                all_plugins.append(
                    np.concatenate([
                        np.zeros(self.test_source_n),
                        detected_residual_on_targetX
                    ])
                )
                all_ifs.append(
                    np.concatenate([
                        correction_term_source,
                        detected_residual_on_targetX
                    ])
                )
                all_detection_masks_plugin.append(np.concatenate([np.zeros(self.test_source_n), target_mask]))
                all_detection_masks_onestep.append(np.concatenate([detection_on_sourceX > 0, target_mask]))
            logging.info("AGGREGATE COND OUTCOME detected proportions on target %s source %s pool %s", np.mean(detection_on_targetX > 0), np.mean(detection_on_sourceX > 0), np.mean(all_detection_masks_onestep[-1]))
            logging.info("AGGREGATE COND OUTCOME onestep %s plugin %s correction source %s", all_ifs[-1][all_detection_masks_onestep[-1].astype(bool)].mean(), all_plugins[-1][all_detection_masks_plugin[-1].astype(bool)].mean(), correction_term_source[detection_on_sourceX > 0].mean())
            logging.info("AGGREGATE COND OUTCOME bias of source loss %s odds %s", (source_loss - exp_loss_source_on_sourceX).mean(), odds_target_x.mean())

        self.agg_detectors_y = detectors

        return {
            'plugin': InferenceResult(
                all_plugins,
                all_detection_masks_plugin
            ),
            'onestep': InferenceResult(
                all_ifs,
                all_detection_masks_onestep
            )}

    def _make_shuffled_data(self, Xdata: np.ndarray, subgroup_mask: np.ndarray, replace: bool = True) -> np.ndarray:
        num_obs = Xdata.shape[0]
        rand_idxs = np.random.choice(num_obs, size=num_obs, replace=replace)
        tildeXs_Xms = Xdata.copy()
        tildeXs_Xms[:, subgroup_mask] = Xdata[rand_idxs][:, subgroup_mask]
        return tildeXs_Xms
    
    def _estimate_density_ustat(self, targetX_train: np.ndarray, source_outcome_model, subgroup_mask: np.ndarray, num_shuffles: int = 10):
        """
        Fits density ratio model for (X_s,X_-s) paired vs (X_s, \tilde{X}_-s) unpaired (i.e. independent), conditional on the tildeX and the original X being in the same bin
        """
        # NOTE: fit density model on all data and not only the detected points to increase number of samples
        targetX_train_detected = targetX_train #self.targetX_train[detector.predict(self.targetX_train) > 0]

        binned_X = self.binning.bin(convert_prob_to_logit(self._get_probability(source_outcome_model, targetX_train_detected)))
        targetX_train_with_bin = np.concatenate([targetX_train_detected, binned_X[:, np.newaxis]], axis=1)
        density_model_ustat = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(targetX_train_detected.shape[1] + 1),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        num_obs = targetX_train_detected.shape[0]

        # create a bunch of shuffled data, i.e. tildeXs_Xms with the correct bins
        tot_num_tilde = 0
        all_tilde_with_bin = []
        for i in range(num_shuffles):
            tildeXs_Xms = self._make_shuffled_data(targetX_train_detected, subgroup_mask)
            binned_tildeX = self.binning.bin(convert_prob_to_logit(self._get_probability(source_outcome_model, tildeXs_Xms)))
            bin_mask = binned_tildeX == binned_X
            tot_num_tilde += bin_mask.sum()
            tilde_with_bin = np.concatenate([tildeXs_Xms, binned_X[:, np.newaxis]], axis=1)
            all_tilde_with_bin.append(tilde_with_bin[bin_mask])
        all_tilde_with_bin = np.concatenate(all_tilde_with_bin, axis=0)
        
        assert tot_num_tilde > 0
        # TODO: remove hack: right now this is here to make sure prevalence of paired vs unpaired is equal
        min_obs = min(num_obs, tot_num_tilde)
        logging.info("TOT NUM TILDE tilde_obs: %d orig obs: %d, min obs: %d", tot_num_tilde, targetX_train_detected.shape[0], min_obs)

        # fit density model
        density_model_ustat.fit(
            np.concatenate([targetX_train_with_bin[:min_obs], all_tilde_with_bin[:min_obs]], axis=0),
            np.concatenate([np.ones(min_obs), np.zeros(min_obs)], axis=0)
        )
        if self.do_grid_search:
            logging.info("selected model density_model_ustat %s %s", density_model_ustat.best_estimator_, density_model_ustat.cv_results_)
        return density_model_ustat
    
    # def _detector_function_covariate_detailed(self, exp_loss, lmbda, odds_x, odds_xms, omega):
    #     """Aggregate corresponds to odds_xms = 1
    #     """
    #     return (self.residual_sign * exp_loss - lmbda) * (odds_x * omega - odds_xms) - self.tolerance * odds_xms
    
    def _compute_risk(self, outcome_model, X, mdl_X):
        '''
        Evaluate source or target risk Z_0(x) or Z_1(x) from the fitted source or target outcome model mu_0(x) or mu_1(x)
        X: features for the outcome model
        mdl_X: features for the ml model being explained
        '''
        prob = self._get_probability(outcome_model, X)
        exp_loss = compute_risk(self.loss_func, mdl_X, prob)
        return exp_loss

    def _estimate_detectors_covariate_detailed(self, source_train, target_train, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask: np.ndarray, cached_exp_loss_odds_source_train: np.ndarray, cached_exp_loss_odds_target_train: np.ndarray):
        # odds_sourceXms = self._get_density_ratio_xs(self.sourceX_train, anti_subgroup_mask)
        # odds_targetXms = self._get_density_ratio_xs(self.targetX_train, anti_subgroup_mask)        
        detectors_omega = []
        for omega in self.candidate_omegas:
            detector, detected_residual, prevalence_source, prevalence_target = self._estimate_detectors_covariate_fixed_omega(source_train, target_train, omega, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask, cached_exp_loss_odds_source_train, cached_exp_loss_odds_target_train)
            # print("omega %.3f detector lambda %.3f detected residual %.3f detection prevalence source %.3f target %.3f" % (omega, detector.min_lambda, detected_residual, prevalence_source, prevalence_target))
            logging.info("omega %.3f detector lambda %.3f detected residual %.3f detection prevalence source %.3f target %.3f", omega, detector.min_lambda, detected_residual, prevalence_source, prevalence_target)
            # if ~np.isnan(detected_residual) and (prevalence_source >= self.min_prevalence) and (prevalence_target >= self.min_prevalence):  # optionally, threshold prevalence in target data
            # Filter when detected residual is nan which happens because of zero detections in source
            if ~np.isnan(detected_residual) and (prevalence_target >= self.min_prevalence):
                detectors_omega.append((detected_residual, detector, omega))
        detectors_omega = sorted(detectors_omega, key=lambda x: -x[0])
        logging.info("COV choosing top %f from %f detectors with highest detected residual", MAX_DETECTORS_COV, len(detectors_omega))
        if len(detectors_omega)==0:
            print("COV detector not found")
        return [(d,o) for (_,d,o) in detectors_omega][:MAX_DETECTORS_COV]

    
    def get_exp_loss_odds_detector(self, data, outcome_model, density_x_model, odds_ratio_Xms_fn, anti_subgroup_mask):
        exp_loss = self._compute_risk(outcome_model, data.X, data.mdl_X)
        odds_x = self._get_density_ratio_x(density_x_model, data.X)
        if odds_ratio_Xms_fn is None:
            odds_xms = 1  # odds_xms is 1 for aggregate
        else:
            odds_xms = get_density_ratio_from_classifier(
                odds_ratio_Xms_fn.predict_proba(data.X[:, anti_subgroup_mask]),
                scale=self.domain_ratio_train,
                eps=self.domain_prob_cutoff
            )
        return {
            'exp_loss': exp_loss,
            'odds_x': odds_x,
            'odds_xms': odds_xms
        }
    
    def _estimate_detectors_covariate_fixed_omega(self, source_train: DataLoader, target_train: DataLoader, omega: float, source_outcome_model, density_x_model, density_ratio_xms_model=None, anti_subgroup_mask: np.ndarray=None, cached_exp_loss_odds_source_train: np.ndarray=None, cached_exp_loss_odds_target_train: np.ndarray = None):
        """cached_exp_loss_odds_source_train: cached exp loss and odds for sourceX_train to reuse for every omega
        """
        detector = DetectorCovariateShift(
            exp_loss_fn=self._compute_risk,
            odds_ratio_X_fn=self._get_density_ratio_x,
            source_outcome_model=source_outcome_model,
            density_x_model=density_x_model,
            odds_ratio_Xms_fn=density_ratio_xms_model,
            residual_sign=self.residual_sign,
            tolerance=self.tolerance,
            domain_ratio_train=self.domain_ratio_train,
            domain_prob_cutoff=self.domain_prob_cutoff,
            anti_subgroup_mask=anti_subgroup_mask
        )

        # Search candidate lambdas for the lambda that minimizes detected residual
        detector.fit(source_train, self.candidate_lambdas, omega, cached_exp_loss_odds_source_train)

        # Evaluate found detector
        detected_residual = detector.predict(source_train, omega, cached_exp_loss_odds_source_train)  # reuse cached exp loss and odds for sourceX_train
        prevalence_source = np.mean(detected_residual > 0)
        detected_residual[detected_residual < 0] = 0  # residual * detector since detector is defined as 1[residual > 0]

        # Normalize residuals to bring it back to original test statistic, in order to select among omegas
        if density_ratio_xms_model is None:
            odds_ratio_sourceXms = 1
        else:
            odds_ratio_sourceXms = cached_exp_loss_odds_source_train['odds_xms']
        weighted_prevalence_source = np.mean((detected_residual > 0) * odds_ratio_sourceXms)
        norm_detected_residual = np.mean(detected_residual) / weighted_prevalence_source
        # print("detected residual min lambda %s" % (detected_residual))
        print("weighted prevalence mean %s norm detected residual %s" % (prevalence_source, norm_detected_residual))

        # Get prevelance in target data E_1[d(x)]
        detected_target = detector.predict(target_train, omega, cached_exp_loss_odds_target_train)
        prevalence_target = np.mean(detected_target > 0)

        return detector, norm_detected_residual, prevalence_source, prevalence_target
        
    def _estimate_detectors_covariate_aggregate(self, source_train, target_train, source_outcome_model, density_x_model, cached_exp_loss_odds_source_train: np.ndarray, cached_exp_loss_odds_target_train: np.ndarray): 
        detectors_omega = []
        for omega in self.candidate_omegas:
            detector, detected_residual, prevalence_source, prevalence_target = self._estimate_detectors_covariate_fixed_omega(source_train, target_train, omega, source_outcome_model, density_x_model, None, None, cached_exp_loss_odds_source_train, cached_exp_loss_odds_target_train)
            logging.info("omega %s detector lambda %s detected residual %s detection proportion source %s target %s", omega, detector.min_lambda, detected_residual, prevalence_source, prevalence_target)
            # Threshold by prevalence of detections in source and target data
            if (prevalence_source >= self.min_prevalence) and (prevalence_target >= self.min_prevalence):
                detectors_omega.append((detected_residual, detector, omega))
        detectors_omega = sorted(detectors_omega, key=lambda x: -x[0])
        logging.info("AGGREGATE COV choosing top %f from %f detectors with highest detected residual", MAX_DETECTORS_COV, len(detectors_omega))
        if len(detectors_omega)==0:
            print("AGGREGATE COV detector not found")
        return [(d,o) for (_,d,o) in detectors_omega][:MAX_DETECTORS_COV]
    
    def _estimate_detectors_cond_outcome_aggregate(self, target_train: DataLoader):
        residual_models = []
        target_loss = self.loss_func(target_train.mdl_X, target_train.Y)
        source_prob_on_targetX = self._get_probability(self.source_outcome_model, target_train.X)
        exp_loss_source_on_targetX = compute_risk(self.loss_func, target_train.mdl_X, source_prob_on_targetX)
        residual = self.residual_sign*(target_loss - exp_loss_source_on_targetX) - self.tolerance
        for detector_base_mdl in DETECTORS:
            detector = sklearn.base.clone(detector_base_mdl)
            detector.fit(
                target_train.X,
                residual
            )
            residual_models.append(detector)
        return residual_models
    
    def _get_detections_cond_outcome(self, targetX: np.ndarray, source_outcome_model, target_outcome_model, y_Xms_model, anti_subgroup_mask: np.ndarray):
        """Detector for outcome test evaluated as a function of the two outcome models
        """
        target_prob_on_targetX = self._get_probability(target_outcome_model, targetX)
        exp_target_loss = compute_risk(self.loss_func, targetX, target_prob_on_targetX)  # X same as mdl_X for outcome test
        source_prob_on_targetX = self._get_probability(source_outcome_model, targetX)
        binned_source_prob_on_targetX = self.binning.bin(convert_prob_to_logit(source_prob_on_targetX))

        prob_on_targetXsW = y_Xms_model.predict_proba(
            np.concatenate([
                targetX[:, anti_subgroup_mask],
                binned_source_prob_on_targetX[:, np.newaxis],
            ], axis=1)
        )[:,1]
        exp_loss_on_targetXms = compute_risk(self.loss_func, targetX, prob_on_targetXsW)
        residual = self.residual_sign * (exp_target_loss - exp_loss_on_targetXms) - self.tolerance
        return residual > 0

    def _estimate_outcome_Xms_source(self, source_train: DataLoader, anti_subgroup_mask: np.ndarray, detections: np.ndarray):
        y_Xms_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(anti_subgroup_mask.sum()+1),
            model_args=OUTCOME_MODEL_ARGS,
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        y_Xms_model.fit(
            np.concatenate([
                source_train.X[:, anti_subgroup_mask],
                detections[:, np.newaxis]
            ], axis=1),
            source_train.Y
        )
        if self.do_grid_search:
            logging.info("selected model y_Xms_model %s %s", y_Xms_model.best_estimator_, y_Xms_model.cv_results_)
        return y_Xms_model
    
    def _estimate_outcome_Xms_bin(self, target_train: DataLoader, source_outcome_model, anti_subgroup_mask: np.ndarray):
        source_prob_on_targetX = self._get_probability(source_outcome_model, target_train.X)
        binned_source_prob_on_targetX = self.binning.bin(convert_prob_to_logit(source_prob_on_targetX))

        y_Xms_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(anti_subgroup_mask.sum()+1),
            model_args=OUTCOME_MODEL_ARGS,
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        y_Xms_model.fit(
            np.concatenate([
                target_train.X[:, anti_subgroup_mask],
                binned_source_prob_on_targetX[:, np.newaxis],
            ], axis=1),
            target_train.Y
        )
        if self.do_grid_search:
            logging.info("selected model y_Xms_model %s %s", y_Xms_model.best_estimator_, y_Xms_model.cv_results_)
        return y_Xms_model
    
    def _get_probability(self, model, X):
        return model.predict_proba(X)[:,1]

    # def _get_source_probability(self, X):
    #     return self.source_data_generator._get_prob(X).flatten()

    def _get_density_ratio_x(self, model, X, scale=None, eps=None):
        if scale is None:
            scale = self.domain_ratio_train
        if eps is None:
            eps = self.domain_prob_cutoff
        odds_x = get_density_ratio_from_classifier(model.predict_proba(X), scale=scale, eps=eps)  # multiply odds by p(a=0)/p(a=1) to get density ratio
        # logging.info("ODDS X %f %f", odds_x.min(), odds_x.max())
        return odds_x
    # def _get_density_ratio_x(self, X):
    #     odds_x = self.density_model_x(X)
    #     logging.info("ODDS X %f %f", odds_x.min(), odds_x.max())
    #     return odds_x
    
    # def _get_density_ratio_xs(self, X, subgroup_mask):
    #     odds_xs = self.density_model_xs(X, subgroup_mask)
    #     logging.info("ODDS Xs %f %f", odds_xs.min(), odds_xs.max())
    #     return odds_xs
    
    def _test_covariate_shift_tolerance_onesided(self, source_train: DataLoader, source_test: DataLoader, target_train: DataLoader, target_test: DataLoader, source_outcome_model, density_x_model, anti_subgroup_mask: np.ndarray):
        """
        Restricted score test for conditional covariate shift
        Returns:
            _type_: _description_
        """
        logging.info("conditional covariate test")
        ## FIT models
        # Initialize density model without conditioning on detections
        # Use it to get detectors and subsequently training data for the correct density and outcome models
        init_density_model_xms = self._estimate_density_model_xs(source_train.X, target_train.X, anti_subgroup_mask, detections_source_target=None)
        # init_density_model_xms = None
        # Cache exp loss and odds to reuse in predict()
        cached_exp_loss_odds_source_train = self.get_exp_loss_odds_detector(source_train, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask)
        cached_exp_loss_odds_target_train = self.get_exp_loss_odds_detector(target_train, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask)

        detectors_omega = self._estimate_detectors_covariate_detailed(source_train, target_train, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask, cached_exp_loss_odds_source_train, cached_exp_loss_odds_target_train)
        logging.info("detectors above min prevalence %s", detectors_omega)

        ## EVALUATE plugin
        # On source train
        # init_odds_on_sourceXms_train = self._get_density_ratio_xs(self.sourceX_train, anti_subgroup_mask)
        
        # On source test
        source_loss = self.loss_func(source_test.mdl_X, source_test.Y)
        source_prob_sourceX = self._get_probability(source_outcome_model, source_test.X)
        exp_loss_source_on_sourceX = compute_risk(self.loss_func, source_test.mdl_X, source_prob_sourceX)
        odds_on_sourceX = self._get_density_ratio_x(density_x_model, source_test.X)
        # init_odds_on_sourceXms = self._get_density_ratio_xs(sourceX, anti_subgroup_mask)
        cached_exp_loss_odds_source_test = self.get_exp_loss_odds_detector(source_test, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask)

        # On target train
        # init_odds_on_targetXms_train = self._get_density_ratio_xs(self.targetX_train, anti_subgroup_mask)
        cached_exp_loss_odds_target_train = self.get_exp_loss_odds_detector(target_train, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask)
        
        # On target test
        source_prob_targetX = self._get_probability(source_outcome_model, target_test.X)
        exp_loss_source_on_targetX = compute_risk(self.loss_func, target_test.mdl_X, source_prob_targetX)
        # init_odds_on_targetXms = self._get_density_ratio_xs(targetX, anti_subgroup_mask)
        cached_exp_loss_odds_target_test = self.get_exp_loss_odds_detector(target_test, source_outcome_model, density_x_model, init_density_model_xms, anti_subgroup_mask)

        all_plugins = []
        all_ifs = []
        all_detection_masks = []
        if len(detectors_omega) == 0:
            all_plugins.append(-np.ones(self.test_source_n+self.test_target_n))
            all_ifs.append(-np.ones(self.test_source_n+self.test_target_n))
            all_detection_masks.append(np.zeros(self.test_source_n + self.test_target_n))
            logging.info("COV detections below threshold %s", self.min_prevalence)
        else:
            for (detector, omega) in detectors_omega:
                logging.info("detector omega %s lambda %s", omega, detector.min_lambda)

                # Detections on train with initialized density model
                init_detection_source_train = detector.predict(source_train, omega, cached_exp_loss_odds_source_train)
                init_detection_target_train = detector.predict(target_train, omega, cached_exp_loss_odds_target_train)
                init_detection_source_target_train = np.concatenate([init_detection_source_train > 0, init_detection_target_train > 0])  # give residuals to density model instead of binary detections
                
                # Refit density model on train set with detections as a feature
                density_model_xms = self._estimate_density_model_xs(source_train.X, target_train.X, anti_subgroup_mask, detections_source_target=init_detection_source_target_train)

                # Fit marginalized outcome model on train set with detections as a feature
                y_Xms_model = self._estimate_outcome_Xms_source(source_train, anti_subgroup_mask, init_detection_source_train > 0)
                
                # # Optionally fit outcome model conditioned on new detections
                # # Recompute detections from the correct density model conditioned on initialized detections
                # odds_on_sourceXms_train = get_density_ratio_from_classifier(
                #     density_model_xms.predict_proba(np.concatenate([
                #         source_train.X[:, anti_subgroup_mask], init_detection_source_train[:, np.newaxis] > 0
                #         ], axis=1)), 
                #     scale=self.domain_ratio_train)  
                # detection_source_train = self._detector_function_covariate_detailed(exp_loss_source_on_sourceX_train, detector_lambda, odds_on_sourceX_train, odds_on_sourceXms_train, detector_omega)
                # y_Xms_model = self._estimate_outcome_Xms_source(source_train, anti_subgroup_mask, detection_source_train)
                
                # Evaluate plugin and onestep on test set with fitted models
                # Get detections with odds from initialized density model
                detection_source = detector.predict(source_test, omega, cached_exp_loss_odds_source_test)
                source_mask = detection_source > 0
                detection_target = detector.predict(target_test, omega, cached_exp_loss_odds_target_test)
                logging.info("detections source %s target %s", detection_source, detection_target)
                target_mask = detection_target > 0

                # Evaluate outcomes and density passing detections feature to the models
                sourceXms_detections = np.concatenate([
                    source_test.X[:, anti_subgroup_mask],
                    detection_source[:, np.newaxis] > 0
                ], axis=1)
                targetXms_detections = np.concatenate([
                    target_test.X[:, anti_subgroup_mask],
                    detection_target[:, np.newaxis] > 0
                ], axis=1)
                # Outcomes
                prob_on_sourceXms = y_Xms_model.predict_proba(sourceXms_detections)[:,1]
                exp_loss_source_on_sourceXms = compute_risk(self.loss_func, source_test.mdl_X, prob_on_sourceXms)
                # exp_loss_source_on_sourceXms = np.ones_like(exp_loss_source_on_sourceXms)
                prob_on_targetXms = y_Xms_model.predict_proba(targetXms_detections)[:,1]
                exp_loss_source_on_targetXms = compute_risk(self.loss_func, target_test.mdl_X, prob_on_targetXms)
                # exp_loss_source_on_targetXms = np.ones_like(exp_loss_source_on_targetXms)
                # Density
                odds_on_sourceXms = get_density_ratio_from_classifier(density_model_xms.predict_proba(sourceXms_detections), scale=np.mean(source_mask)/np.mean(target_mask), eps=self.domain_prob_cutoff)
                # odds_on_sourceXms = self._get_density_ratio_xs(sourceX, anti_subgroup_mask)
                logging.info("ODDS Xms detections min %s max %s value %s", odds_on_sourceXms.min(), odds_on_sourceXms.max(), odds_on_sourceXms)

                logging.info("omega actual %s estimated %s", omega, np.mean(source_mask * odds_on_sourceXms)/np.mean(target_mask))

                # Plugin and IFs
                detection_prevalence_target = sum(target_mask) / (sum(target_mask) + sum(source_mask))
                detection_prevalence_source = 1 - detection_prevalence_target
                all_plugins.append(
                    np.concatenate([
                        (-self.residual_sign * source_loss * odds_on_sourceXms - self.tolerance) / detection_prevalence_source,
                        (self.residual_sign * exp_loss_source_on_targetX) / detection_prevalence_target
                    ])
                )
                all_ifs.append(
                    np.concatenate([
                        (-self.residual_sign * ((source_loss - exp_loss_source_on_sourceXms) * odds_on_sourceXms - (source_loss - exp_loss_source_on_sourceX) * odds_on_sourceX) - self.tolerance) / detection_prevalence_source,
                        (self.residual_sign * (exp_loss_source_on_targetX - exp_loss_source_on_targetXms)) / detection_prevalence_target
                    ])
                )
                all_detection_masks.append(np.concatenate([source_mask, target_mask]))
                logging.info("COV detection proportions source %s target %s", np.mean(source_mask), np.mean(target_mask))
                logging.info("COV exp loss source %s target %s", np.mean(source_loss[source_mask]), np.mean(exp_loss_source_on_targetX[target_mask]))
                logging.info("COV bias in exp loss Xms model source %s", np.mean(source_loss - exp_loss_source_on_sourceXms))
                logging.info("COV odds means Xms %s X %s", odds_on_sourceXms[source_mask].mean(), odds_on_sourceX.mean())
                logging.info("COV plugin %s", exp_loss_source_on_targetX[target_mask].mean() - (source_loss * odds_on_sourceXms)[source_mask].mean())
                logging.info("COV correction term 1 %s", ((source_loss - exp_loss_source_on_sourceX) * odds_on_sourceX)[source_mask].mean())
                logging.info("COV correction term 2 %s", (exp_loss_source_on_sourceXms * odds_on_sourceXms)[source_mask].mean() - exp_loss_source_on_targetXms[target_mask].mean())

        self.detail_detectors_x[anti_subgroup_mask.__str__()] = detectors_omega
        self.detail_detectors_x['correlated_features'] = self.correlated_features

        # TODO: get features with non zero coefficients in source outcome and density models
        features_in_detectors = np.ones(source_test.X.shape[1], dtype=bool)  # assuming all features in correlated features mask

        # Get pvalues from testing whether features used in detector are independent of loss function
        if self.filter_independent_features:
            _, pvalues_correlated_features = get_correlated_features(
                X=source_test.X[:, features_in_detectors], 
                Y=self.loss_func(source_test.mdl_X, source_test.Y), 
                alpha=self.filter_significance_level
            )
        else:
            pvalues_correlated_features = None
        logging.info("pvalues for features in detectors %s", pvalues_correlated_features)

        return {
            'plugin': InferenceResult(
                all_plugins,
                all_detection_masks,
                pvalues_correlated_features
            ),
            'onestep': InferenceResult(
                all_ifs,
                all_detection_masks,
                pvalues_correlated_features
            )
        }

    def _test_cond_outcome_shift_tolerance_onesided(self, target_train: DataLoader, target_test: DataLoader, source_outcome_model, target_outcome_model, anti_subgroup_mask: np.ndarray) -> Dict[str, InferenceResult]:
        """
        Test if shift in performance can be explained by a conditional outcome shift with respect to X_{-s}
        """
        # TODO: remove variable naming with score
        logging.info("conditional outcome test %s", anti_subgroup_mask)

        # FIT models
        y_Xms_model = self._estimate_outcome_Xms_bin(target_train, source_outcome_model, anti_subgroup_mask)
        ustat_classifier = self._estimate_density_ustat(target_train.X, source_outcome_model, ~anti_subgroup_mask)

        # EVALUATE
        target_loss = self.loss_func(target_test.mdl_X, target_test.Y)
        source_prob_on_targetX = self._get_probability(source_outcome_model, target_test.X)
        binned_source_prob_on_targetX = self.binning.bin(convert_prob_to_logit(source_prob_on_targetX))

        prob_on_targetXms = y_Xms_model.predict_proba(
            np.concatenate([target_test.X[:, anti_subgroup_mask], binned_source_prob_on_targetX[:, np.newaxis]], axis=1)
        )[:,1]
        exp_loss_on_targetXms = compute_risk(self.loss_func, target_test.mdl_X, prob_on_targetXms)

        detections = self._get_detections_cond_outcome(target_test.X, source_outcome_model, target_outcome_model, y_Xms_model, anti_subgroup_mask)
        detection_prevalence = np.mean(detections)

        logging.info("COND OUTCOME DETECTIONS PREVALENCE PLUGIN %.3f", detection_prevalence)
        if detection_prevalence <= self.min_prevalence:
            # Nothing detected
            all_scores_plugin = [-np.ones(self.test_source_n + self.test_target_n)]
            all_scores_onestep = [-np.ones(self.test_source_n + self.test_target_n)]
            all_detection_masks_plugin = [np.zeros(self.test_source_n + self.test_target_n)]
            all_detection_masks_onestep = [np.zeros(self.test_source_n + self.test_target_n)]
        else:
            # Something detected
            all_detection_masks_plugin = [np.concatenate([np.zeros(self.test_source_n), detections])]
            all_detection_masks_onestep = [np.concatenate([np.zeros(self.test_source_n), np.ones(self.test_target_n)])]

            # assemble plugin estimator
            all_scores_plugin = [
                np.concatenate([
                    np.zeros(self.test_source_n),
                    (self.residual_sign * (target_loss - exp_loss_on_targetXms) - self.tolerance) * detections
                ])
            ]
            logging.info("plugin result %f", np.mean(all_scores_plugin[0][self.test_source_n:])/detection_prevalence)
            logging.info("bias y_Xms_model %f", (target_loss - exp_loss_on_targetXms).mean())

            # assemble one-step estimator for the numerator
            detections_onestep = detections.copy()
            # ustat_eif1 = exp_loss_on_targetXms * detections_onestep
            ustat_eif2 = np.zeros(target_test.X.shape[0])
            for i in tqdm(range(target_test.X.shape[0])):
                # TODO: currently quite slow
                curr_bin = binned_source_prob_on_targetX[i]
                curr_targetY = target_test.Y[i]
                curr_prob_Xms = prob_on_targetXms[i]

                # hold Xms to the observation i's value
                tildeXs_curr_Xms = target_test.X.copy()
                tildeXs_curr_Xms[:, anti_subgroup_mask] = target_test.X[i, anti_subgroup_mask]
                tildeXs_curr_Xms = self._make_shuffled_data(tildeXs_curr_Xms, ~anti_subgroup_mask, replace=False)
                
                # Filter for tildeX in the same bin and in the subgroup
                source_prob_on_tildeXs_Xms = self._get_probability(source_outcome_model, tildeXs_curr_Xms)
                binned_tildeX = self.binning.bin(convert_prob_to_logit(source_prob_on_tildeXs_Xms))
                keep_mask = (binned_tildeX == curr_bin)
                tildeXs_curr_Xms = tildeXs_curr_Xms[keep_mask]
                tilde_detections = self._get_detections_cond_outcome(tildeXs_curr_Xms, source_outcome_model, target_outcome_model, y_Xms_model, anti_subgroup_mask)
                # assert np.sum(keep_mask) > 1
                if np.sum(keep_mask) > 0:
                    tilde_loss = self.loss_func(tildeXs_curr_Xms, curr_targetY * np.ones(tildeXs_curr_Xms.shape[0]))  # X same as mdl_X for outcome test
                    tildeXs_curr_Xms_with_bin = np.concatenate([tildeXs_curr_Xms, curr_bin * np.ones((tildeXs_curr_Xms.shape[0], 1))], axis=1)
                    tilde_density_ratio = get_density_ratio_from_classifier(ustat_classifier.predict_proba(tildeXs_curr_Xms_with_bin), scale=1, eps=self.ustat_prob_cutoff)
                    # tilde_density_ratio = 1
                    # logging.info("tilde_density_ratio %f %f %f", np.max(tilde_density_ratio), np.mean(tilde_density_ratio), np.min(tilde_density_ratio))

                    exp_tilde_loss = compute_risk(self.loss_func, tildeXs_curr_Xms, curr_prob_Xms * np.ones(tildeXs_curr_Xms.shape[0]))
                    assert tilde_loss.shape == exp_tilde_loss.shape
                    
                    tilde_eif2 = (tilde_loss - exp_tilde_loss) * tilde_detections * tilde_density_ratio
                    ustat_eif2[i] = np.mean(tilde_eif2)
                else:
                    detections_onestep[i] = False
            detection_onestep_prevalence = np.mean(detections_onestep)
            ustat_eif1 = exp_loss_on_targetXms * detections_onestep
            logging.info("COND OUTCOME DETECTIONS PREVALENCE ONESTEP %.3f", detection_onestep_prevalence)
            logging.info("ustat_eif1 %.4f ustat_eif2 %.4f", ustat_eif1.mean()/detection_onestep_prevalence, ustat_eif2.mean()/detection_onestep_prevalence)

            # delta method influence function calculation
            numerator_inf_func_uncentered = (self.residual_sign * (target_loss * detections_onestep - ustat_eif1 - ustat_eif2) - self.tolerance * detections_onestep)
            numerator_inf_func = numerator_inf_func_uncentered - np.mean(numerator_inf_func_uncentered)
            denominator_inf_func = detections_onestep - detection_onestep_prevalence
            decay_estimate = np.mean(numerator_inf_func_uncentered)/detection_onestep_prevalence

            all_scores_onestep = [
                np.concatenate([
                    np.zeros(self.test_source_n),
                    # influence function as a result of the delta method
                    decay_estimate + numerator_inf_func/detection_onestep_prevalence - decay_estimate * denominator_inf_func/detection_onestep_prevalence
                ])
            ]

        self.detail_detectors_y[anti_subgroup_mask.__str__()] = y_Xms_model
        self.detail_detectors_y['correlated_features'] = self.correlated_features

        return {
            'plugin': InferenceResult(
                all_scores_plugin,
                all_detection_masks_plugin,
            ),
            'onestep': InferenceResult(
                all_scores_onestep,
                all_detection_masks_onestep,
            )}

    def test_aggregate(self) -> Tuple[dict, dict, dict]:
        """Covariate and outcome tests
        """
        if self.agg_res_y is not None:
            # if we already have computed these terms
            return self.agg_res_x, self.agg_res_y
        
        st_time = time.time()
        self._prepare_nuisance_models(self.filter_independent_features)
        logging.info("NUISANCE TIME %d", time.time() - st_time)
        
        # Run covariate test on all or correlated features
        st_time = time.time()
        self.agg_res_x = self._test_covariate_aggregate(
            self.source_train_filter,
            self.source_test_filter, 
            self.target_train_filter,
            self.target_test_filter,
            self.source_outcome_filter_model,
            self.density_x_filter_model
        )
        logging.info("aggregate covariate TIME %d", time.time() - st_time)

        st_time = time.time()
        # Run outcome test on all features
        self.agg_res_y = self._test_cond_outcome_aggregate(
            self.target_train,
            self.source_test,
            self.target_test,
            self.source_outcome_model,
            self.density_x_model
        )
        logging.info("aggregate outcome TIME %d", time.time() - st_time)

        logging.info(
            "AGGREGATE DECOMP loss: COVARIATE plug-in %f eif %f; OUTCOME plug-in %f eif %f",
            self.agg_res_x['plugin'].estim,
            self.agg_res_x['onestep'].estim,
            self.agg_res_y['plugin'].estim,
            self.agg_res_y['onestep'].estim,
            )
        return self.agg_res_x, self.agg_res_y

    def test_covariate_detailed(self, subgroup_mask: np.ndarray, one_sided_test: bool) -> dict:
        logging.info("COND COV DECOMP %s", subgroup_mask)
        # subgroup_mask_tuple = tuple(subgroup_mask.tolist())

        self._prepare_nuisance_models(self.filter_independent_features)
        
        # Filter features independent of the loss function
        subgroup_mask_filter = subgroup_mask[self.correlated_features]  # keep only correlated features in mask since input features are filtered
        subgroup_mask_filter_tuple = tuple(subgroup_mask_filter.tolist())
        logging.info("COND COV DECOMP filter features %s", subgroup_mask_filter)

        if subgroup_mask_filter_tuple in self.detail_covariate_res_dict:
            # Check if computed already
            return self.detail_covariate_res_dict[subgroup_mask_filter_tuple]
        
        if np.all(subgroup_mask_filter):
            # EMpty feature subgroup is same as aggregate test
            if self.agg_res_x is None:
                self.agg_res_x, self.agg_res_y = self.test_aggregate()
            return self.agg_res_x
        
        if one_sided_test:
            covariate_res = self._test_covariate_shift_tolerance_onesided(
                self.source_train_filter,
                self.source_test_filter,
                self.target_train_filter,
                self.target_test_filter,
                self.source_outcome_filter_model,
                self.density_x_filter_model,
                ~subgroup_mask_filter)
        else:
            raise NotImplementedError
        
        logging.info(
            "CONDITIONAL COVARIATE test statistic %s: plug-in %f eif %f",
            subgroup_mask_filter,
            covariate_res['plugin'].estim,
            covariate_res['onestep'].estim,
            )
        
        self.detail_covariate_res_dict[subgroup_mask_filter_tuple] = {
                'plugin': covariate_res['plugin'],
                'onestep': covariate_res['onestep'],
        }
        for plot_k, v in self.detail_covariate_res_dict[subgroup_mask_filter_tuple].items():
            logging.info("COND COV TEST %s: %s", plot_k, to_str_inf_res(v))
            print("COND COV TEST %s: %s" % (plot_k, to_str_inf_res(v)))
        return self.detail_covariate_res_dict[subgroup_mask_filter_tuple]
    
    def test_cond_outcome_detailed(self, subgroup_mask: np.ndarray, one_sided_test: bool) -> dict:
        logging.info("COND OUTCOME detailed %s", subgroup_mask)
        subgroup_mask_tuple = tuple(subgroup_mask.tolist())

        self._prepare_nuisance_models(filter_independent_features=False)
        
        if subgroup_mask_tuple in self.detail_cond_outcome_res_dict:
            # Check if computed already
            return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]
        
        if np.all(subgroup_mask):
            # EMpty feature subgroup is same as aggregate test
            if self.agg_res_y is None:
                self.agg_res_x, self.agg_res_y = self.test_aggregate()
            return self.agg_res_y
        
        if one_sided_test:
            cond_outcome_res = self._test_cond_outcome_shift_tolerance_onesided(
                self.target_train,
                self.target_test,
                self.source_outcome_model,
                self.target_outcome_model,
                ~subgroup_mask)
        else:
            raise NotImplementedError
            
        logging.info(
            "CONDITIONAL OUTCOME test statistic %s: plug-in %f eif %f",
            subgroup_mask,
            cond_outcome_res['plugin'].estim,
            cond_outcome_res['onestep'].estim,
            )
    
        self.detail_cond_outcome_res_dict[subgroup_mask_tuple] = {
            'plugin': cond_outcome_res['plugin'],
            'onestep': cond_outcome_res['onestep'],
        }
        for plot_k, v in self.detail_cond_outcome_res_dict[subgroup_mask_tuple].items():
            logging.info("COND OUTCOME TEST %s: %s", plot_k, to_str_inf_res(v))
            print("COND OUTCOME TEST %s: %s" % (plot_k, to_str_inf_res(v)))
        return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]
