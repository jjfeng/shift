import numpy as np
from scipy.stats import multivariate_normal, cauchy

class DataGenerator:
    """
    Base class: generates uniform X
    """
    def __init__(self, beta, intercept, x_mean, nonlinear, source_beta=None, x_source_mean=None, subgroupshift=False, scale=np.array([2,2,2,2]), source_scale=np.array([2,2,2,2])):
        self.beta = beta
        self.x_mean = x_mean.reshape((1,-1))
        self.intercept = intercept
        self.total_p = beta.size
        self.nonlinear = nonlinear
        self.subgroupshift = subgroupshift
        self.source_beta = source_beta
        self.scale = scale
        if x_source_mean is not None:
            self.x_source_mean = x_source_mean.reshape((1,-1))
        self.source_scale = source_scale

    def _get_prob(self, X):
        transform_X = X
        if self.nonlinear:
            transform_X = np.concatenate([
                np.abs(X[:,:1]),
                X[:,1:]
                ], axis=1)
        logit = np.matmul(transform_X, self.beta.reshape((-1,1))) + self.intercept
        return 1/(1 + np.exp(-logit))

    def _get_density_X(self, X):
        return np.logical_and(
            np.all(X >= -self.scale + self.x_mean, axis=1),
            np.all(X <= self.scale + self.x_mean, axis=1),
        ).astype(float)
    
    def _get_density_Xs(self, X, subgroup_mask):
        return np.logical_and(
            np.all(X[:, subgroup_mask] >= -self.scale + self.x_mean[:, subgroup_mask], axis=1),
            np.all(X[:, subgroup_mask] <= self.scale + self.x_mean[:, subgroup_mask], axis=1),
        ).astype(float)
    
    def _generate_X(self, num_obs):
        return (np.random.rand(num_obs, self.total_p) - 0.5) * self.scale * 2 + self.x_mean
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return (np.random.rand(Xms.shape[0], subgroup.sum()) - 0.5) * self.scale * 2 + self.x_mean[:, subgroup]
    
    def _generate_Y(self, X):
        probs = self._get_prob(X)
        y = np.random.binomial(1, probs.flatten(), size=probs.size)
        return y

    def generate(self, num_obs):
        X = self._generate_X(num_obs)
        y = self._generate_Y(X)
        return X, y    
    
class DataGeneratorBox(DataGenerator):
    """Generates X from a box either in positive/negative x1 region and
    Y from a logistic regression model
    """
    
    def _generate_X(self, num_obs):
        print(self.x_mean[:,1:], num_obs)
        X1_pos = np.random.binomial(n=1, p=self.x_mean[0,0], size=(num_obs,1))  # x_mean[0,0] is prob that X1 is positive
        X = np.concatenate([
            (X1_pos - 0.5) * np.random.rand(num_obs, 1) * self.scale * 2,
            (np.random.rand(num_obs, self.total_p-1) - 0.5) * self.scale * 2 + self.x_mean[:,1:]
        ], axis=1)
        return X
    
    def _get_density_X(self, X):
        return self.x_mean[0,0] * np.logical_and(X[:,0]>=0, X[:,0]<=self.scale) + (1-self.x_mean[0,0]) * np.logical_or(X[:,0]<0, X[:,0]>self.scale)
    
    def _get_density_Xs(self, X, subgroup_mask):
        if subgroup_mask[0]:
            return self._get_density_X(X)
        else:
            return super()._get_density_Xs(X, subgroup_mask)
    
class DataGeneratorTree(DataGenerator):
    """
    Generates uniformly distributed X and generates Y from a tree
    """
    def _get_prob(self, X):
        transform_X = X
        if self.nonlinear:
            transform_X = np.concatenate([
                np.abs(X[:,:1]),
                X[:,1:]
                ], axis=1)
        prob = 0.1 * np.ones((transform_X.shape[0],1))
        pos1 = (transform_X[:,0] >= 0) & (transform_X[:,1] >= 0)
        pos2 = (transform_X[:,0] < 0) & (transform_X[:,2] >= 0)
        prob[pos1] = self.beta[0]
        prob[pos2] = self.beta[1]
        return prob

class DataGeneratorMultiNorm(DataGenerator):
    """
    Base class: generates normally distributed X
    """
    
    def _get_density_X(self, X):
        return multivariate_normal.pdf(X, mean=self.x_mean.flatten(), cov=self.scale**2)
    
    def _get_density_Xs(self, X, subgroup_mask):
        if np.all(~subgroup_mask):
            return 1
        else:
            return multivariate_normal.pdf(X[:, subgroup_mask], mean=self.x_mean[:, subgroup_mask].flatten(), cov=self.scale[subgroup_mask]**2)
    
    def _generate_X(self, num_obs):
        return np.random.randn(num_obs, self.total_p) * self.scale + self.x_mean
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return np.random.randn(Xms.shape[0], subgroup.sum()) * self.scale[subgroup] + self.x_mean[:, subgroup]
    
class DataGeneratorMultiNormRestrict(DataGeneratorMultiNorm):
    """
    Shift covariate or outcomes in a box"""
    # lower_interval = -3.5  # for aggregate outcome
    # upper_interval = 3.5
    lower_interval = -4
    upper_interval = 4
    def _get_prob(self, X):
        transform_X = X
        if self.nonlinear:
            transform_X = np.concatenate([
                np.abs(X[:,:1]),
                X[:,1:]
                ], axis=1)
        logit = np.matmul(transform_X, self.beta.reshape((-1,1))) + self.intercept
        box_idx = ((transform_X[:,0] < self.lower_interval) | (transform_X[:,0] > self.upper_interval))
        print("get_prob, proportion of X in box", np.mean(box_idx))
        if self.subgroupshift and ~np.all(self.source_beta==self.beta):  # do not shift for covariate shift only setting
            logit[~box_idx] = np.matmul(transform_X[~box_idx], self.source_beta.reshape((-1,1))) + self.intercept  # keep source logit when x not in subgroup
            print("shift beta %s, source beta %s" % (self.beta, self.source_beta))
        else:
            logit = np.matmul(transform_X, self.beta.reshape((-1,1))) + self.intercept
        return 1/(1 + np.exp(-logit))
    
    def _generate_X(self, num_obs):
        if self.subgroupshift and (~np.all(self.x_source_mean==self.x_mean) or ~np.all(self.source_scale==self.scale)):  # do not shift for outcome shift only setting
            X = np.random.randn(num_obs, self.total_p) * self.source_scale + self.x_source_mean  # from source
            box_idx = ((X[:,0] < self.lower_interval) | (X[:,0] > self.upper_interval))
            print("generate_X, proportion of X in box", np.mean(box_idx))
            X[box_idx] = np.random.randn(sum(box_idx), self.total_p) * self.scale + self.x_mean  # generate covariates from target in the box
            print("shift source mean %s, shift mean %s, source scale %s, shift scale %s" % (self.x_source_mean, self.x_mean, self.source_scale, self.scale))
            return X
        else:
            return np.random.randn(num_obs, self.total_p) * self.scale + self.x_mean
    
    def _get_density_X(self, X):
        raise NotImplementedError
    
    def _get_density_Xs(self, X, subgroup_mask):
        raise NotImplementedError
    
    def _generate_Xs_Xms(self, subgroup_mask, Xms):
        raise NotImplementedError

class DataGeneratorSeqNorm(DataGenerator):
    """
    Base class: generates normally distributed X
    """
    w_scale = 1
    def __init__(self, beta, intercept, x_mean, w_indices, scale, nonlinear):
        self.beta = beta
        self.x_mean = x_mean.reshape((1,-1))
        self.intercept = intercept
        self.total_p = beta.size
        self.nonlinear = nonlinear
        self.scale = scale

    def _generate_X(self, num_obs):
        w = np.random.randn(num_obs, self.num_w_feat) * self.w_scale + self.x_mean[:,:self.num_w_feat]
        x_early_eps = np.random.randn(num_obs, self.total_p-1 - self.num_w_feat) * self.scale
        x_early = x_early_eps + w + self.x_mean[:,self.num_w_feat:-1]
        eps1 = np.random.randn(num_obs, 1) * 0.5 + 0.5 + self.x_mean[:,-1:]
        eps2 = np.random.randn(num_obs, 1) * 0.5 - 0.5 + self.x_mean[:,-1:]
        choice = np.random.choice(2, size=(num_obs, 1), replace=True)
        x_later = -x_early[:,-1:] + eps1 * choice + eps2 * (1 - choice)
        # eps1 = np.random.randn(num_obs, 1) - 1
        # eps2 = np.random.randn(num_obs, 1) + 1
        # choice = x_early[:,-1:] > 0
        # x_later = eps1 * choice + eps2 * (1 - choice)
        return np.concatenate([w, x_early, x_later], axis=1)
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return np.random.randn(Xms.shape[0], subgroup.sum()) * self.scale + self.x_mean[:, subgroup]

    def _get_density_X(self, X):
        raise NotImplementedError()
    
    def _get_density_Xs(self, X, subgroup_mask):
        raise NotImplementedError()
    
class DataGeneratorMultiNormAntiCausal(DataGenerator):
    """
    Base class: generates normally distributed X | Y
    """
    w_scale = 1
    x_scale = 2

    def _get_density_X(self, X, Y):
        print("MEAN", self.x_mean.flatten(), Y * self.beta.flatten())
        return multivariate_normal.pdf(
            X, 
            mean=self.x_mean.flatten() + Y * self.beta.flatten(), 
            cov=self.x_scale**2
        )
    
    def _get_density_Xs(self, X, subgroup_mask, Y):
        if np.all(~subgroup_mask):
            return 1
        else:
            return multivariate_normal.pdf(
                X[:, subgroup_mask], 
                mean=self.x_mean[:, subgroup_mask].flatten() + Y * self.beta[subgroup_mask].flatten(), 
                cov=self.x_scale**2
            )  # Xs and X-s are independent
    
    def _generate_X(self, y):
        # k = 10
        # w = np.random.choice(a=np.arange(k)/k*4, size=(len(y), self.num_w_feat), p=np.ones(k)/k)
        w = np.random.randn(len(y), self.num_w_feat) * self.w_scale + self.x_mean[:,self.w_mask]
        # w = np.random.randn(len(y), self.num_w_feat) * self.w_scale + self.x_mean[:,self.w_mask] + y[:,np.newaxis] * np.repeat(self.beta[self.w_mask][np.newaxis], len(y), axis=0)
        eps = np.random.randn(len(y), self.total_p - self.num_w_feat) * self.x_scale
        x = eps + self.x_mean[:,~self.w_mask] + y[:,np.newaxis] * np.repeat(self.beta[~self.w_mask][np.newaxis], len(y), axis=0)
        xw = np.zeros((len(y), self.total_p))
        xw[:, self.w_mask] = w
        xw[:, ~self.w_mask] = x
        print("DATA w, x, xw", w, x, xw)
        print("PROBABILITY w, xw", self._get_density_W(w, Y=1), self._get_density_X(xw, Y=1), self._get_density_X(xw, Y=0))
        return xw
    
    def _generate_Y(self, num_obs):
        return np.random.binomial(1, p=0.5, size=(num_obs,))
    
    def generate(self, num_obs):
        y = self._generate_Y(num_obs)
        X = self._generate_X(y)
        return X, y    