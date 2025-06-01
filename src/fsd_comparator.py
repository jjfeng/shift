import numpy as np
from sklearn.utils import check_random_state
from fsd import FeatureShiftDetector
from fsd.divergence import FisherDivergence
from fsd.models import GaussianDensity
np.float = float

class FeatureShiftDetectorPvalue(FeatureShiftDetector):
    def detect_and_localize(self, X, Y, random_state=None, return_scores=False):
        """Performs distribution shift detection and localization to features
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical distribution of samples from a reference distribution
        Y : array-like, shape (n_samples, n_features)
            An empirical distribution of samples from the query distribution, i.e. the distribution we want to know if
            it has shifted away from the reference distribution
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        return_scores: bool, optional (default=False)
            If return_scores is True, then the scores of each features will be returned. The default is False.

        Returns
        ----------
        detection : int
            If at least one feature's score is above the detection threshold (i.e. if a feature shift has been detected)
            returns 1 if detected and 0 otherwise.
        attacked_features : array (n_compromised,) or None
            If a detection has occurred, then this will return the indices of the features which are predicted to
            have shifted (i.e. returns the estimated attack set), and if a detection has not occurred then  returns None
        """

        self._check_fitted()
        rng = check_random_state(random_state)
        scores = self.statistic.fit(X, Y).score_features(random_state=rng)
        # Compute pvalues for each feature
        scores_ = np.repeat(scores[np.newaxis,:], self.bootstrap_score_distribution_.shape[0], axis=0)
        pvalues = np.mean(self.bootstrap_score_distribution_ >= scores_, axis=0)
        # Testing for detection of a shift
        if np.any(scores > self.detection_thresholds_):
            detection = 1
            # since shift detection, now localize
            # first we normalize the scores, so we grab the score which has shifted the greatest from it's "null"
            bootstrap_score_means = np.nanmean(self.bootstrap_score_distribution_, axis=0)
            bootstrap_score_std = np.nanstd(self.bootstrap_score_distribution_, axis=0)
            normalized_scores = (scores - bootstrap_score_means) / bootstrap_score_std
            attacked_features = normalized_scores.argsort()[-self.n_compromised:]
        else:
            detection = 0
            attacked_features = None
        if not return_scores:
            return pvalues, detection, attacked_features
        else:
            return pvalues, detection, attacked_features, scores