import json

import pandas as pd
import numpy as np

import scipy.stats as st
from scipy.stats import pearsonr, spearmanr, kendalltau


def coherence_3_workers(
    list_3_workers: list, criterion: str, data: pd.DataFrame, method: str
) -> pd.DataFrame:
    """
    This function returns the correlation between the annotations of 3 workers over a criterion.
    It also checks if the correlation is significant and prints it if not significant.
    :param list_3_workers: list of 3 workers
    :param criterion: criterion to check the correlation
    :param data: dataframe with the data
    :param method: method to compute the correlation
    :return: correlation between the annotations of 3 workers over a criterion
    """
    # annotations for the same worker
    first_worker_criterion = data[data["Worker ID"] == list_3_workers[0]][
        criterion
    ].reset_index(drop=True)
    second_worker_criterion = data[data["Worker ID"] == list_3_workers[1]][
        criterion
    ].reset_index(drop=True)
    third_worker_criterion = data[data["Worker ID"] == list_3_workers[2]][
        criterion
    ].reset_index(drop=True)
    # dataframe with all
    df = pd.DataFrame(
        {
            list_3_workers[0] + "_" + criterion: first_worker_criterion,
            list_3_workers[1] + "_" + criterion: second_worker_criterion,
            list_3_workers[2] + "_" + criterion: third_worker_criterion,
        }
    )
    # check if the correlation is significant
    if method == "pearson":
        if (
            not st.pearsonr(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[1] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[1],
                "over",
                criterion,
                "is significant:",
                st.pearsonr(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[1] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.pearsonr(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.pearsonr(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.pearsonr(
                df[list_3_workers[1] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[1],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.pearsonr(
                    df[list_3_workers[1] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        return df.corr(method="pearson")
    if method == "spearman":
        if (
            not st.spearmanr(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[1] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[1],
                "over",
                criterion,
                "is significant:",
                st.spearmanr(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[1] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.spearmanr(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.spearmanr(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.spearmanr(
                df[list_3_workers[1] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[1],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.spearmanr(
                    df[list_3_workers[1] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        return df.corr(method="spearman")
    if method == "kendall":
        if (
            not st.kendalltau(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[1] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[1],
                "over",
                criterion,
                "is significant:",
                st.kendalltau(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[1] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.kendalltau(
                df[list_3_workers[0] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[0],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.kendalltau(
                    df[list_3_workers[0] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        if (
            not st.kendalltau(
                df[list_3_workers[1] + "_" + criterion],
                df[list_3_workers[2] + "_" + criterion],
            )[1]
            < 0.05
        ):
            print(
                "Correlation between workers",
                list_3_workers[1],
                "and",
                list_3_workers[2],
                "over",
                criterion,
                "is significant:",
                st.kendalltau(
                    df[list_3_workers[1] + "_" + criterion],
                    df[list_3_workers[2] + "_" + criterion],
                )[1]
                < 0.05,
            )
        return df.corr(method="kendall")
    print("Method not found")
    return None


def compute_criteria_correlation(data: pd.DataFrame, correlation: str = "pearson"):
    """
    Compute the correlation between all criteria in the data.
    :param data: the data
    :param correlation: the correlation method to use
    :return: the correlation matrix and the pairs of criteria for which the correlation is not significant
    """
    correlation_criteria = pd.DataFrame(index=data.columns, columns=data.columns)
    not_significant_for_criteria_pairs = set()
    for i, c1 in enumerate(data.columns):
        for j, c2 in enumerate(data.columns):
            if correlation == "pearson":
                correlation_criteria.iloc[i, j] = pearsonr(data[c1], data[c2])[0]
                if not pearsonr(data[c1], data[c2])[1] < 0.05:
                    not_significant_for_criteria_pairs.add((c1, c2))
            elif correlation == "spearman":
                correlation_criteria.iloc[i, j] = spearmanr(data[c1], data[c2])[0]
                if not spearmanr(data[c1], data[c2])[1] < 0.05:
                    not_significant_for_criteria_pairs.add((c1, c2))
            elif correlation == "kendall":
                correlation_criteria.iloc[i, j] = kendalltau(data[c1], data[c2])[0]
                if not kendalltau(data[c1], data[c2])[1] < 0.05:
                    not_significant_for_criteria_pairs.add((c1, c2))
            else:
                print("Correlation not supported")
    print(
        "Correlation not significant for the following criteria pairs:",
        not_significant_for_criteria_pairs,
    )
    return correlation_criteria.astype(float), not_significant_for_criteria_pairs


def system_level_correlation(
    m1: str, m2: str, data: pd.DataFrame, correlation: str = "pearson"
):
    """
    Compute the correlation between two metrics at the system level.
    :param m1: the first metric
    :param m2: the second metric
    :param data: the data
    :param correlation: the correlation method to use
    :return: the correlation
    """
    data_m1 = np.array([json.loads(l) for l in data[m1].tolist()]).T
    data_m2 = np.array([json.loads(l) for l in data[m2].tolist()]).T
    not_significant_for_metric_pairs = set()
    if correlation == "pearson":
        corr = pearsonr(np.mean(data_m1, axis=0), np.mean(data_m2, axis=0))
        if corr[1] >= 0.05:
            not_significant_for_metric_pairs.add((m1, m2))
    elif correlation == "spearman":
        corr = spearmanr(np.mean(data_m1, axis=0), np.mean(data_m2, axis=0))
        if corr[1] >= 0.05:
            not_significant_for_metric_pairs.add((m1, m2))
    elif correlation == "kendall":
        corr = kendalltau(np.mean(data_m1, axis=0), np.mean(data_m2, axis=0))
        if corr[1] >= 0.05:
            not_significant_for_metric_pairs.add((m1, m2))
    else:
        print("Correlation {} not supported.".format(correlation))
        corr = None
    return corr[0], not_significant_for_metric_pairs


def get_all_system_level_correlation(
    data: pd.DataFrame, correlation_type: str = "kendall"
):
    """
    Compute the correlation between all metrics at the system level.
    :param data: the data
    :param correlation_type: the correlation method to use
    :return: the correlation matrix
    """
    system_level_correlation_metrics_human = pd.DataFrame(
        index=data.columns[1:7], columns=data.columns[7:]
    )
    not_significant_for_metric_pairs = set()
    for i, m1 in enumerate(data.columns[1:7]):
        for j, m2 in enumerate(data.columns[7:]):
            system_level_correlation_metrics_human.iloc[
                i, j
            ] = system_level_correlation(m1, m2, data, correlation_type)[0]
            not_significant_for_metric_pairs = not_significant_for_metric_pairs.union(
                system_level_correlation(m1, m2, data, correlation_type)[1]
            )
    print(
        "Correlation not significant for the following metric pairs:",
        not_significant_for_metric_pairs,
    )
    return system_level_correlation_metrics_human.astype(float)


def story_level_correlation(
    m1: str, m2: str, data: pd.DataFrame, correlation: str = "pearson"
):
    """
    Compute the correlation between two metrics at the story level.
    :param m1: the first metric
    :param m2: the second metric
    :param data: the data
    :param correlation: the correlation method to use
    :return: the correlation
    """

    data_m1 = np.array([json.loads(l) for l in data[m1].tolist()]).T
    data_m2 = np.array([json.loads(l) for l in data[m2].tolist()]).T
    if correlation == "pearson":
        return np.mean(
            [
                pearsonr(data_m1[i], data_m2[i])[0]
                for i in range(data_m1.shape[0])
                if not np.isnan(pearsonr(data_m1[i], data_m2[i])[0])
            ]
        )
    elif correlation == "spearman":
        return np.mean(
            [
                spearmanr(data_m1[i], data_m2[i])[0]
                for i in range(data_m1.shape[0])
                if not np.isnan(spearmanr(data_m1[i], data_m2[i])[0])
            ]
        )
    elif correlation == "kendall":
        return np.mean(
            [
                kendalltau(data_m1[i], data_m2[i])[0]
                for i in range(data_m1.shape[0])
                if not np.isnan(kendalltau(data_m1[i], data_m2[i])[0])
            ]
        )
    else:
        print("Correlation {} not supported.".format(correlation))
    return correlation


def get_all_story_level_correlation(
    data: pd.DataFrame, correlation_type: str = "kendall"
):
    """
    Compute the correlation between all metrics at the story level.
    :param data: the data
    :return: the correlation matrix
    """
    story_level_correlation_metrics_human = pd.DataFrame(
        index=data.columns[1:7], columns=data.columns[7:]
    )
    for i, m1 in enumerate(data.columns[1:7]):
        for j, m2 in enumerate(data.columns[7:]):
            story_level_correlation_metrics_human.iloc[i, j] = story_level_correlation(
                m1, m2, data, correlation_type
            )
    return story_level_correlation_metrics_human.astype(float)


def metrics_ranking(
    data: pd.DataFrame,
    correlation_level: str,
    correlation: str,
    list_criteria: list,
    k_best: any = None,
):
    """
    Compute the ranking of the metrics according to the bordas method.
    :param correlation_level: the level of correlation to use whether it is at the system level or at the story level
    :param correlation: the correlation method to use
    :param list_criteria: the list of criteria to use
    :param data: the data
    :param k_best: the number of best metrics to return if None all the metrics are returned
    :return: the ranking of the metrics
    """
    df = pd.DataFrame(index=data.columns[7:], columns=data.columns[1:7])
    if correlation_level == "system":
        corr = get_all_system_level_correlation(data, correlation)
    elif correlation_level == "story":
        corr = get_all_story_level_correlation(data, correlation)
    else:
        print("this level:", correlation_level, "is not supported")
    for criterion in list_criteria:
        df.loc[:, criterion] = corr.loc[criterion, :].rank(ascending=False)
    df["bordas_rank"] = df.sum(axis=1)
    return df.sort_values(by="bordas_rank", ascending=True).head(k_best)


def ASG_ranking_human_criteria(
    data_scores: pd.DataFrame, list_criteria: list, k_best: any = None
):
    """
    Compute text generators ranking according to the bordas method.
    :param data_scores: the data
    :param list_criteria: the list of criteria to use
    :param k_best: the number of best text generators to return if None all the text generators are returned
    :return: the ranking of the text generators
    """
    df = pd.DataFrame(index=data_scores.index, columns=data_scores.columns)
    for criterion in list_criteria:
        df.loc[:, criterion] = data_scores.loc[:, criterion].rank(ascending=False)
    df["bordas_rank_human_criteria"] = df.sum(axis=1)
    sorted_df = df.sort_values(by="bordas_rank_human_criteria", ascending=True)
    if k_best is None:
        return sorted_df
    return sorted_df.head(k_best)
