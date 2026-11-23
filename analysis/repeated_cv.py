"""Repeated nested cross-validation and validation-only permutation importance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

import analysis

DIRECTORY = Path(__file__).resolve().parent
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
PARAM_GRID = dict(n_estimators=[500], max_depth=[None, 6],
                  max_features=["sqrt", 1.0], min_samples_leaf=[2, 3],
                  min_samples_split=[2])
MODELS = analysis.MODEL_COLUMNS
MODEL_NAMES = analysis.MODEL_NAMES
FEATURE_NAMES = analysis.FEATURE_NAMES
FEATURE_COLUMNS = analysis.FEATURE_COLUMNS
FACTORS = analysis.FACTORS
TRIPLES = analysis.TRIPLES
PHYSICAL_COLUMNS = set("NOPQRSTUVWX")


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(native(value), indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_plan(cases, repeats):
    plan = []
    for r in range(repeats):
        folds = []
        outer = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED + r)
        for f, (train, validation) in enumerate(outer.split(np.arange(cases))):
            inner_seed = 10000 + SEED + 100 * r + f
            inner = KFold(INNER_FOLDS, shuffle=True, random_state=inner_seed)
            splits = list(inner.split(train))
            if np.intersect1d(train, validation).size:
                raise ValueError("Outer training and validation indices overlap.")
            for inner_train, inner_validation in splits:
                if (np.intersect1d(train[inner_train], validation).size
                        or np.intersect1d(train[inner_validation], validation).size):
                    raise ValueError("An outer validation case entered inner cross-validation.")
            folds.append(dict(outer_fold=f + 1, training_indices=train.tolist(),
                validation_indices=validation.tolist(), inner_seed=inner_seed,
                permutation_seed=20000 + SEED + 100 * r + f,
                inner_splits=[dict(training_positions=a.tolist(), validation_positions=b.tolist())
                              for a, b in splits]))
        seen = sorted(index for fold in folds for index in fold["validation_indices"])
        if seen != list(range(cases)):
            raise ValueError("Every case must occur once in each outer validation partition.")
        plan.append(dict(repeat=r + 1, outer_seed=SEED + r, folds=folds))
    return plan


def model_units(columns):
    columns = np.asarray(columns)
    groups = [(FEATURE_NAMES[j], (int(j),)) for j in columns]
    position = {int(j): i for i, j in enumerate(columns)}
    numeric = [(FEATURE_NAMES[j], int(j), i) for i, j in enumerate(columns)]
    conceptual = []
    for column in FACTORS:
        indices = tuple(j for j, value in enumerate(FEATURE_COLUMNS) if value == column)
        if indices[0] not in position:
            continue
        if len(indices) == 1:
            group = position[indices[0]]
        else:
            group = len(groups)
            groups.append((column, indices))
        conceptual.append((column, group))
    return groups, numeric, conceptual


UNIT_MAPS = [model_units(columns) for columns in MODELS]


def predict_in_chunks(model, values, maximum_rows=50000):
    result = np.empty(len(values))
    for start in range(0, len(values), maximum_rows):
        stop = min(start + maximum_rows, len(values))
        result[start:stop] = model.predict(values[start:stop])
    return result


def permuted_fold_predictions(model, validation_x, columns, groups, orders):
    draws, cases = orders.shape
    if cases != len(validation_x) or not np.all(np.sort(orders, axis=1) == np.arange(cases)):
        raise ValueError("Invalid within-validation-fold permutation indices.")
    columns = np.asarray(columns)
    perturbed = np.broadcast_to(validation_x,
        (len(groups), draws, cases, len(columns))).copy()
    for group_index, (_, indices) in enumerate(groups):
        for index in indices:
            local = int(np.flatnonzero(columns == index)[0])
            perturbed[group_index, :, :, local] = validation_x[orders, local]
    predicted = predict_in_chunks(model, perturbed.reshape(-1, len(columns)))
    return predicted.reshape(len(groups), draws, cases)


def rank_information(values, top=3):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.empty(0), np.empty(0)
    limit = min(top, len(values))
    credit = np.array([np.clip((limit - np.sum(values > value)) / np.sum(values == value), 0, 1)
                       for value in values])
    if not np.isclose(credit.sum(), limit, rtol=0, atol=1e-12):
        raise ValueError("Top-three inclusion credits have an invalid total.")
    return rankdata(-values, method="average"), credit


def fitted_outer_model(x, y, columns, fold, repeat, model_index, jobs, grid,
                       cache=None, signature=None, resume=False):
    saved_path = (cache / f"repeat_{repeat:02d}_fold_{fold['outer_fold']:02d}_model_{model_index + 1:02d}.joblib"
                  if cache is not None else None)
    if resume and saved_path is not None and saved_path.is_file():
        saved = joblib.load(saved_path)
        if saved["signature"] != signature:
            raise ValueError("The resume cache does not match the data and analysis protocol.")
        return saved["model"], saved["selected"], saved["search_rows"], True
    train = np.array(fold["training_indices"])
    inner = [(np.array(item["training_positions"]), np.array(item["validation_positions"]))
             for item in fold["inner_splits"]]
    search = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1), grid,
        scoring="neg_mean_squared_error", cv=inner, n_jobs=jobs, refit=True,
        return_train_score=False, error_score="raise")
    search.fit(x[train][:, columns], y[train])
    selected = dict(repeat=repeat, outer_fold=fold["outer_fold"], Model=model_index + 1,
        train_cases=len(train), validation_cases=len(fold["validation_indices"]),
        inner_seed=fold["inner_seed"], best_inner_CV_MSE=float(-search.best_score_),
        **native(search.best_params_))
    scores = []
    for i, settings in enumerate(search.cv_results_["params"]):
        scores.append(dict(repeat=repeat, outer_fold=fold["outer_fold"], Model=model_index + 1,
            setting_index=i + 1, rank=int(search.cv_results_["rank_test_score"][i]),
            **native(settings), mean_inner_CV_MSE=float(-search.cv_results_["mean_test_score"][i]),
            **{f"inner_fold_{j + 1}_MSE": float(-search.cv_results_[f"split{j}_test_score"][i])
               for j in range(INNER_FOLDS)}))
    if saved_path is not None:
        temporary = saved_path.with_suffix(".tmp")
        joblib.dump(dict(signature=signature, model=search.best_estimator_,
                         selected=selected, search_rows=scores), temporary, compress=3)
        temporary.replace(saved_path)
    return search.best_estimator_, selected, scores, False


def calculate_repeat(x, y, plan, jobs, permutations, grid=None,
                     cache=None, signature=None, resume=False):
    grid = PARAM_GRID if grid is None else grid
    cases = len(y)
    oof = np.full((len(MODELS), cases), np.nan)
    perturbed = [np.full((len(units[0]), permutations, cases), np.nan) for units in UNIT_MAPS]
    seen = np.zeros((len(MODELS), cases), dtype=int)
    selected, searches = [], []
    for fold in plan["folds"]:
        validation = np.array(fold["validation_indices"])
        rng = np.random.RandomState(fold["permutation_seed"])
        orders = np.stack([rng.permutation(len(validation)) for _ in range(permutations)])
        for m, columns in enumerate(MODELS):
            started = time.monotonic()
            model, settings, scores, reused = fitted_outer_model(x, y, columns, fold,
                plan["repeat"], m, jobs, grid, cache, signature, resume)
            validation_x = x[validation][:, columns]
            oof[m, validation] = model.predict(validation_x)
            seen[m, validation] += 1
            perturbed[m][:, :, validation] = permuted_fold_predictions(
                model, validation_x, columns, UNIT_MAPS[m][0], orders)
            selected.append(settings)
            searches.extend(scores)
            action = "resumed fit" if reused else "fitted"
            print(f"Repeat {plan['repeat']:02d}, fold {fold['outer_fold']}, model {m + 1}: "
                  f"{action}; {time.monotonic() - started:.1f} s", flush=True)
    if not np.all(seen == 1) or not np.isfinite(oof).all() or any(not np.isfinite(a).all() for a in perturbed):
        raise ValueError("Each case needs exactly one finite OOF prediction per model and repeat.")
    return oof, perturbed, selected, searches


def pooled_metrics(y, prediction):
    errors = prediction - y
    mse = np.mean(errors ** 2)
    return dict(R2=float(1 - mse / np.var(y)), RMSE_log1p=float(np.sqrt(mse)),
                MAE_log1p=float(np.mean(abs(errors))))


def describe(values, prefix):
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return {f"{prefix}_median": float(median), f"{prefix}_Q1": float(q1),
            f"{prefix}_Q3": float(q3), f"{prefix}_min": float(values.min()),
            f"{prefix}_max": float(values.max())}


def scope_ranks(means, columns):
    output = [dict(all_unit_count=len(means)) for _ in means]
    scopes = {"all": list(range(len(means))),
              "built_environment": [i for i, col in enumerate(columns) if col not in "CD"],
              "physical": [i for i, col in enumerate(columns) if col in PHYSICAL_COLUMNS]}
    for scope, indices in scopes.items():
        ranks, credit = rank_information(means[indices])
        lookup = {j: i for i, j in enumerate(indices)}
        for i, record in enumerate(output):
            if scope != "all":
                record[f"{scope}_unit_count"] = len(indices)
            at = lookup.get(i)
            record[f"rank_{scope}"] = None if at is None else float(ranks[at])
            record[f"top3_{scope}_credit"] = None if at is None else float(credit[at])
    return output


def importance_rows(repeat, model, y, oof, perturbed, unit):
    sst = np.sum((y - y.mean()) ** 2)
    base_sse = np.sum((oof - y) ** 2)
    sse = np.sum((perturbed - y[None, None, :]) ** 2, axis=-1)
    delta_r2 = (sse - base_sse) / sst
    delta_rmse = np.sqrt(sse / len(y)) - np.sqrt(base_sse / len(y))
    _, numeric, conceptual = UNIT_MAPS[model]
    if unit == "numeric":
        mapped = [(key, FEATURE_COLUMNS[j], group) for key, j, group in numeric]
    else:
        mapped = [(column, column, group) for column, group in conceptual]
    means = np.array([delta_r2[group].mean() for _, _, group in mapped])
    ranks = scope_ranks(means, [column for _, column, _ in mapped])
    rows = []
    for i, (key, column, group) in enumerate(mapped):
        label = FACTORS[column]
        if unit == "numeric" and column in TRIPLES:
            label += f" (component {key.rsplit('_', 1)[1]})"
        identity = {"numeric_feature": key} if unit == "numeric" else {"column": key}
        rows.append(dict(repeat=repeat, Model=model + 1, Feature_set=MODEL_NAMES[model],
            **identity, Factor=label, delta_R2=float(means[i]),
            delta_RMSE_log1p=float(delta_rmse[group].mean()),
            permutation_SD_delta_R2=float(delta_r2[group].std(ddof=1)), **ranks[i]))
    return rows


def summarize_performance(rows):
    output = []
    for m, name in enumerate(MODEL_NAMES, 1):
        selected = [row for row in rows if row["Model"] == m]
        record = dict(Model=m, Feature_set=name, repeats=len(selected))
        for key in ("R2", "RMSE_log1p", "MAE_log1p"):
            record.update(describe([row[key] for row in selected], key))
        output.append(record)
    return output


def summarize_importance(rows, unit):
    key_name = "numeric_feature" if unit == "numeric" else "column"
    groups = {}
    for row in rows:
        groups.setdefault((row["Model"], row[key_name]), []).append(row)
    output = []
    for (_, key), values in groups.items():
        first = values[0]
        record = {name: first[name] for name in ("Model", "Feature_set", key_name, "Factor")}
        record["repeats"] = len(values)
        for name in ("delta_R2", "delta_RMSE_log1p", "permutation_SD_delta_R2"):
            record.update(describe([row[name] for row in values], name))
        for scope in ("all", "built_environment", "physical"):
            record[f"{scope}_unit_count"] = first[f"{scope}_unit_count"]
            ranks = [row[f"rank_{scope}"] for row in values if row[f"rank_{scope}"] is not None]
            credits = [row[f"top3_{scope}_credit"] for row in values if row[f"top3_{scope}_credit"] is not None]
            if ranks:
                record.update(describe(ranks, f"rank_{scope}"))
            else:
                record.update({f"rank_{scope}_{stat}": None for stat in ("median", "Q1", "Q3", "min", "max")})
            record[f"top3_{scope}_frequency"] = None if not credits else float(np.mean(credits))
        output.append(record)
    for model in range(1, len(MODELS) + 1):
        selected = [row for row in output if row["Model"] == model]
        ranks, _ = rank_information([row["delta_R2_median"] for row in selected])
        for row, rank in zip(selected, ranks):
            row["rank_by_median_delta_R2"] = float(rank)
    return sorted(output, key=lambda row: (row["Model"], -row["delta_R2_median"]))


def write_outputs(output, performance, numeric, conceptual, predictions, assignments, fits, searches):
    numeric_summary = summarize_importance(numeric, "numeric")
    conceptual_summary = summarize_importance(conceptual, "conceptual")
    files = [("performance_by_repeat.csv", performance),
             ("performance_summary.csv", summarize_performance(performance)),
             ("numeric_importance_by_repeat.csv", numeric),
             ("numeric_importance_summary.csv", numeric_summary),
             ("conceptual_importance_by_repeat.csv", conceptual),
             ("conceptual_importance_summary.csv", conceptual_summary),
             ("OOF_predictions.csv", predictions),
             ("outer_fold_assignments.csv", assignments),
             ("selected_parameters.csv", fits), ("inner_CV_results.csv", searches)]
    for name, rows in files:
        analysis.write_csv(output / name, rows)
    priorities = []
    for unit, summary, key_name in (("numeric", numeric_summary, "numeric_feature"),
                                    ("conceptual", conceptual_summary, "column")):
        for key in ("C", "D", "Q", "P"):
            row = next(row for row in summary if row["Model"] == 3 and row[key_name] == key)
            rest = {name: value for name, value in row.items() if name != key_name}
            priorities.append(dict(ranking_unit=unit, unit_key=key, **rest))
    analysis.write_csv(output / "priority_importance_Model3.csv", priorities)


def run(args):
    if args.repeats < 1 or args.permutations < 2 or args.jobs == 0:
        raise ValueError("Use at least one repeat, two permutations, and nonzero jobs.")
    if args.resume and args.cache is None:
        raise ValueError("--resume requires an explicit --cache directory.")
    started = time.monotonic()
    numeric, _, _, coordinates, x, y, _ = analysis.read_data(args.data, args.sheet)
    if x.shape != (55, 30) or np.var(y) <= 0:
        raise ValueError("Expected 55 cases, 30 numeric inputs and nonconstant responses.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve() if args.cache is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    plan = split_plan(len(y), args.repeats)
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "openpyxl", "joblib")}
    protocol = dict(workbook=args.data.name, sheet=args.sheet, records=len(numeric), cases=len(y),
        repeats=args.repeats, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS,
        outer_seeds=[SEED + r for r in range(args.repeats)], forest_seed=SEED,
        inner_seed_rule="10000 + 42 + 100 * repeat_index + fold_index; indices start at zero",
        permutation_seed_rule="20000 + 42 + 100 * repeat_index + fold_index; indices start at zero",
        parameter_grid=PARAM_GRID, parameter_combinations_per_search=8,
        search_score="neg_mean_squared_error", outer_training_searches=args.repeats * OUTER_FOLDS * len(MODELS),
        permutation_draws_per_unit_per_repeat=args.permutations,
        importance_unit="Pooled 55-case OOF delta_R2; mean across permutation draws within each repeat",
        secondary_importance_unit="Pooled perturbed RMSE_log1p minus baseline RMSE_log1p",
        interval="Empirical Q1-Q3 across repeat estimates; not a confidence interval",
        ranking="Descending repeat-mean delta_R2; average ties and fractional top-three boundary credit",
        top3_frequency_unit="Proportion of repeat inclusion credits, from zero to one",
        numeric_all_pool_sizes=[28, 2, 30], numeric_built_pool_sizes=[28, 0, 28],
        numeric_physical_pool_sizes=[19, 0, 19],
        conceptual_all_pool_sizes=[20, 2, 22], conceptual_built_pool_sizes=[20, 0, 20],
        conceptual_physical_pool_sizes=[11, 0, 11],
        unique_perturbation_groups=[len(units[0]) for units in UNIT_MAPS], versions=versions)
    signature_payload = dict(protocol=protocol, workbook_digest=digest(args.data),
                             script_digest=digest(Path(__file__)),
                             helper_digest=digest(Path(analysis.__file__)))
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    save_json(output / "nested_splits.json", plan)
    performance, numeric_rows, conceptual_rows = [], [], []
    prediction_rows, assignments, fits, searches = [], [], [], []
    all_oof, all_perturbed, all_fold_ids = [], [[] for _ in MODELS], []
    for repeat_plan in plan:
        repeat = repeat_plan["repeat"]
        checkpoint = cache / f"repeat_{repeat:02d}_results.npz" if cache is not None else None
        details_path = cache / f"repeat_{repeat:02d}_details.json" if cache is not None else None
        if args.resume and checkpoint is not None and checkpoint.is_file() and details_path.is_file():
            details = json.loads(details_path.read_text(encoding="utf-8"))
            if details["signature"] != signature:
                raise ValueError("The repeat cache does not match the data and analysis protocol.")
            with np.load(checkpoint, allow_pickle=False) as stored:
                oof = stored["oof"]
                perturbed = [stored[f"permuted_{m}"] for m in range(len(MODELS))]
            selected, search = details["selected"], details["search_rows"]
            print(f"Repeat {repeat:02d}: resumed completed repeat", flush=True)
        else:
            oof, perturbed, selected, search = calculate_repeat(x, y, repeat_plan, args.jobs,
                args.permutations, cache=cache, signature=signature, resume=args.resume)
            if checkpoint is not None:
                temporary = checkpoint.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, oof=oof,
                                    **{f"permuted_{m}": values for m, values in enumerate(perturbed)})
                temporary.replace(checkpoint)
                save_json(details_path, dict(signature=signature, selected=selected, search_rows=search))
        fits.extend(selected)
        searches.extend(search)
        fold_ids = np.empty(len(y), dtype=int)
        for fold in repeat_plan["folds"]:
            fold_ids[fold["validation_indices"]] = fold["outer_fold"]
        assignments.extend(dict(repeat=repeat, group_index=g, A=coordinates[g], outer_fold=int(fold_ids[g]))
                           for g in range(len(y)))
        for m in range(len(MODELS)):
            metrics = pooled_metrics(y, oof[m])
            performance.append(dict(repeat=repeat, outer_seed=repeat_plan["outer_seed"],
                Model=m + 1, Feature_set=MODEL_NAMES[m], cases=len(y), **metrics))
            prediction_rows.extend(dict(repeat=repeat, Model=m + 1, group_index=g, A=coordinates[g],
                outer_fold=int(fold_ids[g]), observed_log1p_B=float(y[g]),
                predicted_log1p_B=float(oof[m, g]), residual=float(y[g] - oof[m, g])) for g in range(len(y)))
            numeric_rows.extend(importance_rows(repeat, m, y, oof[m], perturbed[m], "numeric"))
            conceptual_rows.extend(importance_rows(repeat, m, y, oof[m], perturbed[m], "conceptual"))
            all_perturbed[m].append(perturbed[m])
        all_oof.append(oof)
        all_fold_ids.append(fold_ids)
        write_outputs(output, performance, numeric_rows, conceptual_rows,
                      prediction_rows, assignments, fits, searches)
        print(f"Repeat {repeat:02d}: complete pooled metrics and importance saved", flush=True)
    np.savez_compressed(output / "OOF_prediction_arrays.npz", observed_log1p_B=y,
        baseline_oof=np.stack(all_oof), outer_fold_ids=np.stack(all_fold_ids),
        coordinates=np.asarray(coordinates),
        **{f"permuted_Model{m + 1}": np.stack(values) for m, values in enumerate(all_perturbed)},
        **{f"permutation_groups_Model{m + 1}": np.asarray([key for key, _ in units[0]])
           for m, units in enumerate(UNIT_MAPS)})
    print(f"Completed {args.repeats} repeats in {time.monotonic() - started:.1f} s. "
          f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DIRECTORY / "analysis_data.xlsx")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "results" / "repeated_cv")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--cache", type=Path, help="Optional directory for local resumable fits and repeat arrays.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching files in --cache.")
    run(parser.parse_args())
