"""Random Forest analysis with training-only grid search and held-out evaluation."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import openpyxl
from scipy.spatial.distance import cdist
from scipy.stats import norm, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split

SEED = 42
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
TRIPLES = set("ORWX")
FACTORS = {
    "C": "Residential walkability", "D": "Occupational walkability", "E": "Rail Number",
    "F": "Average dining price", "G": "Average retail price", "H": "Average Service price",
    "I": "Average Entertainment price", "J": "Dining ratio", "K": "Retail ratio",
    "L": "Service ratio", "M": "Entertainment ratio", "N": "Slab transparency",
    "O": "Color of the ceiling", "P": "Enclosure degree", "Q": "Wall transparency",
    "R": "Color of the wall", "S": "Natural elements", "T": "Artificial lighting",
    "U": "Resting spaces; Outdoor seating areas", "V": "Signage Visual Prominence",
    "W": "Color of the floor", "X": "Overall color",
}
FEATURE_NAMES = []
FEATURE_COLUMNS = []
for _column in FACTORS:
    _labels = [f"{_column}_{i}" for i in (1, 2, 3)] if _column in TRIPLES else [_column]
    FEATURE_NAMES.extend(_labels)
    FEATURE_COLUMNS.extend([_column] * len(_labels))
MODEL_COLUMNS = [np.arange(2, 30), np.arange(2), np.arange(30)]
MODEL_NAMES = ["Built-Environment-Only", "Mobile-Signaling-Only", "Combined"]
GRID = dict(n_estimators=[500, 800], max_depth=[None, 6], max_features=["sqrt", 1.0],
            min_samples_leaf=[2, 3], min_samples_split=[2, 5])
METRIC_NAMES = ["R2", "RMSE_log1p", "MAE_log1p", "Bootstrap_R2_mean",
                "Bootstrap_percentile_2_5", "Bootstrap_percentile_97_5"]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_data(path, sheet):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook[sheet].iter_rows(max_col=24, values_only=True))
    finally:
        workbook.close()
    if len(rows) < 2 or rows[0][4] != "Rail Number":
        raise ValueError("Expected the A-X schema, with Rail Number in column E.")
    raw = rows[1:]
    if any(row[0] is None for row in raw):
        raise ValueError("Every data row must contain its coordinate identifier in A.")
    coordinates = list(dict.fromkeys(row[0] for row in raw))
    if len(coordinates) != 55:
        raise ValueError("This analysis requires 55 distinct coordinate-level cases.")
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}
    groups = np.array([lookup[row[0]] for row in raw])
    numeric = np.empty((len(raw), 31))
    for i, row in enumerate(raw):
        numeric[i, 0] = float(row[1])
        position = 1
        for column in FACTORS:
            value = row[ord(column) - ord("A")]
            if column in TRIPLES:
                components = np.array([float(v) for v in NUMBER.findall(str(value))])
                if len(components) != 3 or not np.all((components >= 0) & (components <= 1)):
                    raise ValueError(f"Invalid three-component value at {column}{i + 2}.")
                if abs(components.sum() - 1) > 1e-10:
                    raise ValueError(f"Components must sum to one at {column}{i + 2}.")
                numeric[i, position:position + 3] = components
                position += 3
            else:
                numeric[i, position] = float(value)
                position += 1
    if not np.isfinite(numeric).all() or np.any(numeric[:, :3] < 0):
        raise ValueError("The input must contain finite values and nonnegative B-D.")
    for group in range(55):
        if any(len(np.unique(numeric[groups == group, j])) != 1 for j in range(3)):
            raise ValueError("B-D must be constant within each coordinate.")
    aggregate = np.stack([numeric[groups == group].mean(0) for group in range(55)])
    x = aggregate[:, 1:].copy()
    x[:, :2] = np.log1p(x[:, :2])
    x, y = np.round(x, 10), np.round(np.log1p(aggregate[:, 0]), 10)
    lonlat = np.array([[float(v) for v in re.findall(r"\d+\.\d+", str(a))] for a in coordinates])
    if lonlat.shape != (55, 2):
        raise ValueError("Coordinates must contain decimal longitude and latitude.")
    xy = np.column_stack([
        (lonlat[:, 0] - lonlat[:, 0].mean()) * 111320 * np.cos(np.deg2rad(lonlat[:, 1].mean())),
        (lonlat[:, 1] - lonlat[:, 1].mean()) * 111320,
    ])
    return numeric, aggregate, groups, coordinates, x, y, xy


def performance(y, predictions):
    squared = (predictions - y) ** 2
    mse = squared.mean()
    result = [1 - mse / np.var(y), np.sqrt(mse), abs(predictions - y).mean()]
    ids = np.random.RandomState(SEED).randint(0, len(y), (500, len(y)))
    sampled_y = y[ids]
    denominator = np.sum((sampled_y - sampled_y.mean(1)[:, None]) ** 2, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("A bootstrap resample has zero response variance.")
    bootstrap = 1 - np.sum((predictions[ids] - sampled_y) ** 2, axis=1) / denominator
    return np.r_[result, bootstrap.mean(), np.quantile(bootstrap, [.025, .975])]


def spatial_weights(xy, k):
    distances = cdist(xy, xy)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :k]
    weights = np.zeros_like(distances)
    weights[np.arange(len(xy))[:, None], neighbours] = 1 / k
    return weights


def moran(residuals, weights):
    centered = residuals - residuals.mean()
    n = len(centered)
    s0 = weights.sum()
    denominator = centered @ centered
    observed = n / s0 * (centered @ weights @ centered) / denominator
    expected = -1 / (n - 1)
    s1 = .5 * np.sum((weights + weights.T) ** 2)
    s2 = np.sum((weights.sum(0) + weights.sum(1)) ** 2)
    variance = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n * n - 1) * s0 * s0) - expected ** 2
    z = (observed - expected) / np.sqrt(variance)
    p = 2 * norm.sf(abs(z))
    rng = np.random.RandomState(SEED)
    draws = np.stack([rng.permutation(centered) for _ in range(9999)])
    simulated = n / s0 * np.einsum("bi,ij,bj->b", draws, weights, draws) / denominator
    permutation_p = (1 + np.sum(abs(simulated - expected) >= abs(observed - expected))) / 10000
    return dict(I=float(observed), expected_I=float(expected), z=float(z), p=float(p),
                p_permutation_two_sided=float(permutation_p))


def factor_correlations(numeric, groups, x, y):
    records = []
    for column, name in FACTORS.items():
        indices = [i for i, value in enumerate(FEATURE_COLUMNS) if value == column]
        if column in TRIPLES:
            component_rows = numeric[:, np.array(indices) + 1]
            log_components = np.log(component_rows, out=np.zeros_like(component_rows), where=component_rows > 0)
            row_entropy = -np.sum(component_rows * log_components, axis=1) / np.log(3)
            values = np.array([row_entropy[groups == g].mean() for g in range(55)])
        else:
            values = x[:, indices[0]]
        if np.all(values == values[0]):
            coefficient, p = np.nan, np.nan
        else:
            coefficient, p = pearsonr(values, y)
        records.append(dict(column=column, Factor=name, pearson_r=float(coefficient), p_value=float(p)))
    probabilities = np.array([record["p_value"] for record in records])
    order = np.argsort(probabilities)
    order = order[np.isfinite(probabilities[order])]
    adjusted = np.full(len(records), np.nan)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(
        (probabilities[order] * len(records) / np.arange(1, len(order) + 1))[::-1])[::-1])
    for record, value in zip(records, adjusted):
        record["p_adj_fdr_bh"] = float(value)
        record["significant"] = bool(value < .05)
    return records


def importances(models, x, y, test):
    rng = np.random.RandomState(SEED)
    orders = np.stack([rng.permutation(len(test)) for _ in range(100)])
    factor_groups = [(column, [j for j, value in enumerate(FEATURE_COLUMNS) if value == column])
                     for column in FACTORS]
    factor_results, actual_results = {}, []
    for model_index, columns in enumerate(MODEL_COLUMNS):
        values = x[test][:, columns]
        baseline = models[model_index].predict(values)
        baseline_sse = np.sum((baseline - y[test]) ** 2)
        sst = np.sum((y[test] - y[test].mean()) ** 2)
        included = [(column, indices) for column, indices in factor_groups if indices[0] in columns]
        sets = included + [(FEATURE_NAMES[j], [j]) for j in columns]
        shuffled = np.broadcast_to(values, (len(sets), 100, len(test), len(columns))).copy()
        for k, (_, indices) in enumerate(sets):
            for j in indices:
                local = int(np.flatnonzero(columns == j)[0])
                shuffled[k, :, :, local] = values[orders, local]
        predictions = models[model_index].predict(shuffled.reshape(-1, len(columns)))
        predictions = predictions.reshape(len(sets), 100, len(test))
        draws = (np.sum((predictions - y[test]) ** 2, axis=-1) - baseline_sse) / sst
        means = draws.mean(-1)
        for k, (column, _) in enumerate(included):
            factor_results[model_index, column] = float(means[k])
        component_means = means[len(included):]
        ranks = np.argsort(np.argsort(-component_means, kind="stable"), kind="stable") + 1
        built = [k for k, j in enumerate(columns) if FEATURE_COLUMNS[j] not in "CD"]
        physical = [k for k, j in enumerate(columns) if FEATURE_COLUMNS[j] in "NOPQRSTUVWX"]
        built_ranks = {k: i + 1 for i, k in enumerate(sorted(built, key=lambda k: -component_means[k]))}
        physical_ranks = {k: i + 1 for i, k in enumerate(sorted(physical, key=lambda k: -component_means[k]))}
        for k, j in enumerate(columns):
            column = FEATURE_COLUMNS[j]
            name = FACTORS[column]
            if column in TRIPLES:
                name += f" (component {FEATURE_NAMES[j].rsplit('_', 1)[1]})"
            actual_results.append(dict(Model=model_index + 1, numeric_feature=FEATURE_NAMES[j], Factor=name,
                importance_mean_delta_R2=float(component_means[k]),
                permutation_SD=float(draws[len(included) + k].std(ddof=1)), rank_all=int(ranks[k]),
                all_numeric_features=len(columns), rank_built_environment=built_ranks.get(k),
                built_environment_numeric_features=len(built), rank_physical=physical_ranks.get(k),
                physical_numeric_features=len(physical)))
    return factor_results, actual_results


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    numeric, _, groups, _, x, y, xy = read_data(args.data, args.sheet)
    train, test = train_test_split(np.arange(55), test_size=.2, random_state=SEED)
    inner_cv = KFold(5, shuffle=True, random_state=SEED)
    models, parameters, search_records, table14 = [], [], [], []
    for m, columns in enumerate(MODEL_COLUMNS):
        search = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1), GRID,
                              scoring="neg_mean_squared_error", cv=inner_cv,
                              n_jobs=args.jobs, error_score="raise", refit=True,
                              return_train_score=False)
        search.fit(x[train][:, columns], y[train])
        selected = search.best_params_.copy()
        parameters.append(selected)
        models.append(search.best_estimator_)
        table14.append(dict(Model=m + 1, Feature_set=MODEL_NAMES[m],
                            selection="Training-only GridSearchCV", **selected,
                            inner_CV_mean_MSE=float(-search.best_score_)))
        for settings, score, rank in zip(search.cv_results_["params"],
                                        search.cv_results_["mean_test_score"],
                                        search.cv_results_["rank_test_score"]):
            search_records.append(dict(Model=m + 1, **settings,
                                       inner_CV_mean_MSE=float(-score), rank=int(rank)))
        print(f"Model {m + 1}: {len(columns)} numeric features; parameters {selected}", flush=True)
    table13, table15, oof = [], [], []
    outer_diagnostic_cv = KFold(5, shuffle=True, random_state=SEED)
    for m, columns in enumerate(MODEL_COLUMNS):
        prediction = models[m].predict(x[test][:, columns])
        metrics = performance(y[test], prediction)
        record = dict(Model=m + 1, Feature_set=MODEL_NAMES[m], Sample_size=len(y),
                      train_cases=len(train), test_cases=len(test),
                      No_features=len(columns), Random_seed=SEED)
        record.update({name: float(value) for name, value in zip(METRIC_NAMES, metrics)})
        table13.append(record)
        estimator = RandomForestRegressor(**parameters[m], random_state=SEED, n_jobs=1)
        predictions = cross_val_predict(estimator, x[:, columns], y,
                                        cv=outer_diagnostic_cv, n_jobs=args.jobs)
        oof.append(predictions)
        table15.append(dict(Model=m + 1, k=4,
                            **moran(y - predictions, spatial_weights(xy, 4))))
    table15.append(dict(Model=3, k=5, **moran(y - oof[2], spatial_weights(xy, 5))))
    table16 = factor_correlations(numeric, groups, x, y)
    joint, component = importances(models, x, y, test)
    for record in table16:
        for m in range(3):
            record[f"Model{m + 1}_joint_factor_delta_R2"] = joint.get((m, record["column"]))
    for filename, records in [("Table13.csv", table13), ("Table14.csv", table14),
                              ("Table15.csv", table15), ("Table16.csv", table16),
                              ("Actual_feature_importance.csv", component),
                              ("GridSearch_results.csv", search_records)]:
        write_csv(output / filename, records)
    print(f"Analysis saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=SCRIPT_DIRECTORY / "analysis_data.xlsx",
                        help="Input A-X workbook; default: analysis_data.xlsx beside this script.")
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIRECTORY / "results" / "tables",
                        help="Directory for CSV tables; default: results/tables beside this script.")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Parallel jobs for grid search and OOF fitting; default: 4.")
    run(parser.parse_args())
