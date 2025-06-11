"""Detector class for covariate tests
"""
import time
import logging
import numpy as np
from common import get_density_ratio_from_classifier

class DetectorCovariateShift():
    def __init__(self, exp_loss_fn, odds_ratio_X_fn, source_outcome_model, density_x_model, odds_ratio_Xms_fn, residual_sign, tolerance, domain_ratio_train, domain_prob_cutoff, anti_subgroup_mask):
        self.exp_loss_fn = exp_loss_fn
        self.odds_ratio_X_fn = odds_ratio_X_fn
        self.source_outcome_model = source_outcome_model
        self.density_x_model = density_x_model
        self.odds_ratio_Xms_fn = odds_ratio_Xms_fn  # function from (X, mask) to density ratio value
        self.residual_sign = residual_sign
        self.tolerance = tolerance
        self.anti_subgroup_mask = anti_subgroup_mask
        self.domain_ratio_train = domain_ratio_train
        self.domain_prob_cutoff = domain_prob_cutoff
        
        self.min_lambda = None  # stores lambda found in fit

    def _get_exp_loss_odds(self, data):
        exp_loss = self.exp_loss_fn(self.source_outcome_model, data.X, data.mdl_X)
        odds_x = self.odds_ratio_X_fn(self.density_x_model, data.X, scale=self.domain_ratio_train, eps=self.domain_prob_cutoff)
        if self.odds_ratio_Xms_fn is None:
            odds_xms = 1  # odds_xms is 1 for aggregate
        else:
            odds_xms = get_density_ratio_from_classifier(
                self.odds_ratio_Xms_fn.predict_proba(data.X[:, self.anti_subgroup_mask]),
                scale=self.domain_ratio_train,
                eps=self.domain_prob_cutoff
            )
        return exp_loss, odds_x, odds_xms
    
    def fit(self, data, candidate_lambdas, omega, cached_exp_loss_odds=None):
        if len(candidate_lambdas.shape) == 1:
            candidate_lambdas = candidate_lambdas[:, np.newaxis]  # new axis to have numpy evaluate on each lambda efficiently
        
        detected_residual_lambdas = self._evaluate(data, candidate_lambdas, omega, cached_exp_loss_odds)  # outputs (num lambdas * num samples) array
        detected_residual_lambdas[detected_residual_lambdas < 0] = 0  # optimal choice of detector is to clip the negative residuals at 0
        mean_detected_residual = np.mean(detected_residual_lambdas, axis=1)  # TODO: mean on all source data or on detection[i,:] > 0?

        min_lambda_idx = np.argmin(mean_detected_residual)
        self.min_lambda = candidate_lambdas[min_lambda_idx].item()

    def _evaluate(self, data, lmbda, omega, cached_exp_loss_odds):
        if cached_exp_loss_odds is not None:  # use cached output from fit()
            assert len(cached_exp_loss_odds['exp_loss']) == data.shape[0], "cached_exp_loss_odds does not match input"
            exp_loss = cached_exp_loss_odds['exp_loss']
            odds_x = cached_exp_loss_odds['odds_x']
            odds_xms = cached_exp_loss_odds['odds_xms']
        else:
            exp_loss, odds_x, odds_xms = self._get_exp_loss_odds(data)
        return (self.residual_sign * exp_loss - lmbda) * (odds_x * omega - odds_xms) - self.tolerance * odds_xms
    
    def predict(self, data, omega, cached_exp_loss_odds=None):
        assert self.min_lambda is not None, "fit not called"
        # return np.ones(X.shape[0])
        return self._evaluate(data, self.min_lambda, omega, cached_exp_loss_odds)
    
    def get_features(self):
        """Get features used in the detector
        Should be features in source_outcome_model and density_x_model
        """
        raise NotImplementedError
