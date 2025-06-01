"""
Shift explainer wrappers and base classes
"""
import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from common import *

class InferenceResult:
    """
    Store influence function or plugin evaluation
    feature_independence_pvalue_list is None for outcome tests
    For covariate tests, store pvalues for independence tests for each feature used in detector
    Returns max test statistic across detectors
    """
    def __init__(self, influence_func_list, mask_list, feature_independence_pvalue_list=None):
        self.influence_func_list = influence_func_list
        self.mask_list = [mask.astype(bool) for mask in mask_list]
        self.feature_independence_pvalue_list = feature_independence_pvalue_list
        assert len(mask_list) == len(influence_func_list)

    @property
    def estim(self):
        estims = [np.mean(ic[mask]) for ic, mask in zip(self.influence_func_list, self.mask_list) if mask.sum() > 0]
        if len(estims) == 0:
            return 0
        else:
            return np.max(estims)

def to_str_inf_res(inf_res: InferenceResult) -> str:
    return f"Inference res Estim: %f" % (inf_res.estim)

class ConstantModel:
    def __init__(self, val):
        self.val = val
    def predict(self, testX):
        return self.val * np.ones(testX.shape[0])

class BaseShiftExplainer:
    do_grid_search = False
    is_oracle = True

    def _test_covariate_aggregate(self) -> Dict[str, InferenceResult]:
        raise NotImplementedError()

    def _test_cond_outcome_aggregate(self) -> Dict[str, InferenceResult]:
        raise NotImplementedError()

    def _get_source_probability(self, X):
        raise NotImplementedError()

    def _get_target_probability(self, X):
        raise NotImplementedError()

    def _make_max_feats_list(self, num_feats, min_num_feats=1) -> list:
        max_feats = list(range(min_num_feats, num_feats + 1, max(1, (num_feats + 1)//4)))
        if (len(max_feats) == 0) or (max_feats[-1] != num_feats):
            return max_feats + [None]
        else:
            return max_feats

    def test_aggregate(self) -> Tuple[dict, dict, dict]:
        raise NotImplementedError()

    def test_covariate_detailed(self, subgroup_mask: np.ndarray) -> dict[str, InferenceResult]:
        raise NotImplementedError()
    
    def test_cond_outcome_detailed(
        self, subgroup_mask: np.ndarray
    ) -> Dict[str, InferenceResult]:
        raise NotImplementedError()


class ExplainerInference:
    """Runs aggregate and detailed decomposition for the given list of subsets in combos
    """
    def __init__(
        self,
        shift_explainer: BaseShiftExplainer,
        num_obs: int,
        num_p: int,
        combos: list,
        detailed_lst: list[str] = [COND_COV_STR, COND_OUTCOME_STR],
        do_aggregate: bool = False,
        one_sided_test: bool = False,
        significance_level: float = 0.05,
        num_bootstrap: int = 10000,
    ):
        self.shift_explainer = shift_explainer
        self.num_p = num_p
        self.total_p = shift_explainer.target_loader.total_p
        self.num_obs = num_obs
        self.do_aggregate = do_aggregate
        self.detailed_lst = detailed_lst
        self.do_detailed = (COND_COV_STR in self.detailed_lst) or (COND_OUTCOME_STR in self.detailed_lst)
        self.combos = combos
        self.one_sided_test = one_sided_test
        self.significance_level = significance_level
        self.num_bootstrap = num_bootstrap

        self.agg_res_x = {}
        self.agg_res_y = {}

        self.detailed_covariate_lst = []
        self.detailed_cond_outcome_lst = []

    def run_tests(self):
        if self.do_aggregate:
            self.agg_res_x, self.agg_res_y = self.shift_explainer.test_aggregate()

        if self.do_detailed:
            self.detailed_cond_outcome_lst = []
            self.detailed_covariate_lst = []
            for subgroup_mask in self.combos:
                expanded_mask = np.ones(self.total_p, dtype=bool) 
                # fills in cells in subgroup
                for feature in range(len(expanded_mask)):
                    # obtains group number from a dictionary
                    group_num = self.shift_explainer.target_loader.vdict[feature]
                    # update expanded mask with what's in feature subgroup
                    expanded_mask[feature] = subgroup_mask[group_num]
                logging.info("COND OUTCOME SUBGROUP GROUP %s FEATURE %s", subgroup_mask, expanded_mask)
                if COND_OUTCOME_STR in self.detailed_lst:
                    inf_res = self.shift_explainer.test_cond_outcome_detailed(
                        expanded_mask, self.one_sided_test
                    )
                    self.detailed_cond_outcome_lst.append(inf_res)
                if COND_COV_STR in self.detailed_lst:
                    inf_res = self.shift_explainer.test_covariate_detailed(
                        expanded_mask, self.one_sided_test
                    )
                    self.detailed_covariate_lst.append(inf_res)

    def _run_bootstrap_test(
        self,
        inf_results: List[InferenceResult],
        level: str,
        decomp: str,
        vars: List[str],
        est: str,
        one_sided_test=False,
        significance_level = 0.05,
        n_bootstrap:int = 100000,
    ) -> pd.DataFrame:
        """Get pvalues for McEE test of loss change by multiplier bootstrap
        When we have a composite null hypothesis in case of covariate test,
        then returns maximum of the pvalues from the McEE test and tests for independence
        of each feature used in the detector from the loss function"""
        critical_values = []
        pvalues_composite = []
        pvalues_mcee = []
        decisions = []  # NOTE: made by critical values for McEE test, not valid for composite
        for inf_res in inf_results:
            bootstrap_dist = get_multiplier_bootstrap_dist(inf_res.influence_func_list, inf_res.mask_list, n_bootstrap=n_bootstrap)
            if one_sided_test:
                critical_value = np.quantile(bootstrap_dist, q = 1 - significance_level)
                critical_values.append(critical_value)
                decisions.append(inf_res.estim > critical_value)
                pval_mcee_test = np.mean(bootstrap_dist >= inf_res.estim)
            else:
                critical_value = np.quantile(bootstrap_dist, q = 1 - significance_level/2)
                critical_values.append(critical_value)
                decisions.append(np.abs(inf_res.estim) > critical_value)
                pval_mcee_test = np.mean(np.abs(bootstrap_dist) >= np.abs(inf_res.estim))
            pvalues_mcee.append(pval_mcee_test)
            if inf_res.feature_independence_pvalue_list is None:
                pvalues_composite.append(pval_mcee_test)
            else:
                pvalues_composite.append(max(
                    pval_mcee_test, max(inf_res.feature_independence_pvalue_list)
                ))  # pvalue for the composite null hypothesis in covariate tests

        return pd.DataFrame(
            {
                "value": [inf_res.estim for inf_res in inf_results],
                "critical_value": critical_values,
                # "decision": decisions,  # compute decisions from pvalues
                "pvalue": pvalues_composite,
                "pvalue_mcee": pvalues_mcee,
                "pvalue_independence": [max(inf_res.feature_independence_pvalue_list) if inf_res.feature_independence_pvalue_list is not None else None for inf_res in inf_results],
                "level": level,
                "decomp": decomp,
                "vars": vars,
                "est": est,
            }
        )

    def get_aggregate_res_x(self, plot_key: str = "onestep", significance_level=0.05, n_bootstrap=10000):
        return self._run_bootstrap_test(
            [self.agg_res_x[plot_key]],
            one_sided_test=self.one_sided_test,
            decomp="agg",
            level="agg",
            vars="X",
            est=plot_key,
            significance_level=significance_level,
            n_bootstrap=n_bootstrap,
        )

    def get_aggregate_res_y(self, plot_key: str = "onestep", significance_level=0.05, n_bootstrap=10000):
        return self._run_bootstrap_test(
            [self.agg_res_y[plot_key]],
            one_sided_test=self.one_sided_test,
            decomp="agg",
            level="agg",
            vars="Y",
            est=plot_key,
            significance_level=significance_level,
            n_bootstrap=n_bootstrap,
        )
    
    def get_detailed_res_cond_outcome(
        self, plot_key: str = "onestep", significance_level=0.05, n_bootstrap=10000,
    ) -> pd.DataFrame:
        if COND_OUTCOME_STR not in self.detailed_lst:
            return None

        tstat_inf_res = self._run_bootstrap_test(
            [res_dict[plot_key] for res_dict in self.detailed_cond_outcome_lst],
            one_sided_test=self.one_sided_test,
            level="detail",
            decomp=COND_OUTCOME_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
            significance_level=significance_level,
            n_bootstrap=n_bootstrap,
        )

        return tstat_inf_res

    def get_detailed_res_covariate(
        self, plot_key: str = "onestep", significance_level=0.05, n_bootstrap=10000,
    ) -> dict:
        if COND_COV_STR not in self.detailed_lst:
            return None

        tstat_inf_res = self._run_bootstrap_test(
            [res_dict[plot_key] for res_dict in self.detailed_covariate_lst],
            one_sided_test=self.one_sided_test,
            level="detail",
            decomp=COND_COV_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
            significance_level=significance_level,
            n_bootstrap=n_bootstrap,
        )

        return tstat_inf_res

    def summary(self) -> pd.DataFrame:
        logging.info("SUMMARY num bootstrap %s", self.num_bootstrap)
        df = None
        if self.do_aggregate:
            # Collate aggregate results
            for agg_plot_key in self.agg_res_y.keys():
                agg_res_x = self.get_aggregate_res_x(agg_plot_key, significance_level=self.significance_level, n_bootstrap=self.num_bootstrap)
                agg_res_y = self.get_aggregate_res_y(agg_plot_key, significance_level=self.significance_level, n_bootstrap=self.num_bootstrap)
                agg_df = pd.concat([agg_res_x, agg_res_y])
                df = pd.concat([df, agg_df]) if df is not None else agg_df

        # Collate detailed results
        if COND_OUTCOME_STR in self.detailed_lst:
            detail_plot_keys = self.detailed_cond_outcome_lst[0].keys()
            for detail_plot_key in detail_plot_keys:
                res_cond_outcome_df = self.get_detailed_res_cond_outcome(
                    detail_plot_key, significance_level=self.significance_level, n_bootstrap=self.num_bootstrap
                )
                df = (
                    pd.concat([df, res_cond_outcome_df])
                    if df is not None
                    else res_cond_outcome_df
                )
        if COND_COV_STR in self.detailed_lst:
            detail_plot_keys = self.detailed_covariate_lst[0].keys()
            for detail_plot_key in detail_plot_keys:
                res_covariate_df = self.get_detailed_res_covariate(
                    detail_plot_key, significance_level=self.significance_level, n_bootstrap=self.num_bootstrap
                )
                df = (
                    pd.concat([df, res_covariate_df])
                    if df is not None
                    else res_covariate_df
                )
        return df.reset_index(drop=True)
