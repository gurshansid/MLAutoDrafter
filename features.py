# File to get all the features for the postion feature

import numpy as np
import pandas as pd


def _normalize_feature(
    val: float, col: str, feature_means: pd.Series, feature_stds: pd.Series
) -> float:
    if col in feature_means.index:
        mean = feature_means[col]
        std = feature_stds[col]
        return (val - mean) / std if std > 0 else 0.0
    return val / 100.0


def get_qb_features(player_row, feature_means, feature_stds):
    feature_cols = [
        "fantasy_adp",
        "avg_points_per_game",
        "avg_passing_yards",
        "avg_passing_tds",
        "avg_rushing_yards",  
        "avg_rushing_tds",
    ]

    feats = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            feats.append(_normalize_feature(val, col, feature_means, feature_stds))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def get_rb_features(player_row, feature_means, feature_stds):
    feature_cols = [
        "fantasy_adp",
        "avg_points_per_game",
        "avg_rushing_yards",
        "avg_rushing_tds",
        "avg_receptions",  
        "avg_receiving_yards",
        "avg_receiving_tds",
    ]

    feats = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            feats.append(_normalize_feature(val, col, feature_means, feature_stds))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def get_wr_features(player_row, feature_means, feature_stds):
    feature_cols = [
        "fantasy_adp",
        "avg_points_per_game",
        "avg_targets",  
        "avg_receptions",
        "avg_receiving_yards",
        "avg_receiving_tds",
    ]

    feats = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            feats.append(_normalize_feature(val, col, feature_means, feature_stds))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def get_te_features(player_row, feature_means, feature_stds):
    feature_cols = [
        "fantasy_adp",
        "avg_points_per_game",
        "avg_targets",
        "avg_receptions",
        "avg_receiving_yards",
        "avg_receiving_tds",
    ]

    feats = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            feats.append(_normalize_feature(val, col, feature_means, feature_stds))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)