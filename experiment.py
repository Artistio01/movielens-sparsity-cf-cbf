"""
Эксперимент: влияние разреженности на коллаборативную фильтрацию (CF)
и контентный подход (CBF) на MovieLens 100K.

Запуск (Colab):
    !pip install scikit-surprise
    !python experiment.py --seeds 10

Данные: u.data / u.item оригинального MovieLens 100K
(https://grouplens.org/datasets/movielens/100k/) либо ml.inter / ml.item (формат RecBole).
Параметр --data задаёт папку с файлами.
"""
import argparse, time, warnings
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNWithMeans, BaselineOnly

warnings.filterwarnings("ignore")

Q_LIST = [1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02]   # доли сохраняемых обучающих оценок
K_TOP = 10                                    # K для Precision@K / Recall@K
REL_THR = 4                                   # оценка >= 4 считается релевантной
BETA = 10.0                                   # демпфирование пользовательского смещения
GRID_KNN = [5, 10, 20, 50]
GRID_SVD = [(20, 0.02), (20, 0.1), (50, 0.02), (50, 0.1)]
GRID_LAMBDA = [0.1, 0.25, 0.5, 1.0, 2.0]


# ------------------------------------------------------------------ данные
def _finish(r, genre_names, flags_by_item):
    """Приводит идентификаторы к 0..m-1, 0..n-1 и строит матрицу жанров X."""
    u_ids = np.sort(r["u"].unique())
    i_ids = np.sort(list(flags_by_item.keys()))
    u_map = {u: k for k, u in enumerate(u_ids)}
    i_map = {i: k for k, i in enumerate(i_ids)}
    r = r.copy()
    r["u"] = r["u"].map(u_map)
    r["i"] = r["i"].map(i_map)
    X = np.zeros((len(i_ids), len(genre_names)), dtype=np.float64)
    for i, vec in flags_by_item.items():
        X[i_map[i]] = vec
    return r.reset_index(drop=True), X, genre_names


def load_data(data_dir="."):
    """Понимает два формата MovieLens 100K:
    1) оригинал GroupLens: u.data (user item rating time, табуляция) и u.item (через |);
    2) формат RecBole: ml.inter и ml.item (с заголовком)."""
    import os
    if os.path.exists(os.path.join(data_dir, "u.data")):
        r = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t", header=None, names=["u", "i", "r", "t"])[["u", "i", "r"]]
        it = pd.read_csv(os.path.join(data_dir, "u.item"), sep="|", header=None, encoding="latin-1")
        names = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                 "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                 "Thriller", "War", "Western"]
        flags = {int(row[0]): row[5:24].astype(float).values for _, row in it.iterrows()}
        return _finish(r, names, flags)
    r = pd.read_csv(os.path.join(data_dir, "ml.inter"), sep="\t")
    r.columns = ["u", "i", "r", "t"]
    r = r[["u", "i", "r"]]
    it = pd.read_csv(os.path.join(data_dir, "ml.item"), sep="\t")
    it.columns = ["i", "title", "year", "cls"]
    names = sorted({g for c in it["cls"].fillna("unknown") for g in str(c).split()})
    flags = {}
    for _, row in it.iterrows():
        v = np.zeros(len(names))
        for g in str(row["cls"]).split():
            v[names.index(g)] = 1.0
        flags[int(row["i"])] = v
    return _finish(r, names, flags)


# ------------------------------------------------------------------ вспомогательное
def user_bias(train, m, mu):
    s = np.zeros(m)
    c = np.zeros(m)
    np.add.at(s, train["u"].values, train["r"].values - mu)
    np.add.at(c, train["u"].values, 1.0)
    return s / (c + BETA), c


def fallback_pred(u, bu, cnt, mu):
    """Запасной прогноз: mu + b_u, если пользователь известен; иначе mu."""
    return mu + np.where(cnt[u] > 0, bu[u], 0.0)


def surprise_predict(algo, df_eval, bu, cnt, mu):
    """Прогноз Surprise; где модель не может предсказать - запасной вариант."""
    preds = algo.test(list(zip(df_eval["u"].values, df_eval["i"].values, df_eval["r"].values)))
    est = np.array([p.est for p in preds])
    imp = np.array([p.details.get("was_impossible", False) for p in preds])
    if imp.any():
        est[imp] = fallback_pred(df_eval["u"].values[imp], bu, cnt, mu)
    return np.clip(est, 1, 5)


def to_trainset(df):
    data = Dataset.load_from_df(df[["u", "i", "r"]], Reader(rating_scale=(1, 5)))
    return data.build_full_trainset()


def rmse_mae(y, p):
    e = y - p
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def rank_metrics(df_test, pred, rng, k=K_TOP):
    """Precision@k, Recall@k по оценённым пользователем тестовым фильмам."""
    d = pd.DataFrame({"u": df_test["u"].values, "r": df_test["r"].values,
                      "p": pred + rng.uniform(0, 1e-9, size=len(pred))})
    d = d.sort_values(["u", "p"], ascending=[True, False])
    d["rank"] = d.groupby("u").cumcount()
    d["rel"] = (d["r"] >= REL_THR).astype(int)
    g = d.groupby("u")
    n_items = g["rel"].size()
    n_rel = g["rel"].sum()
    hit = d[d["rank"] < k].groupby("u")["rel"].sum().reindex(n_items.index).fillna(0)
    ok_p = n_items >= k
    ok_r = ok_p & (n_rel > 0)
    prec = float((hit[ok_p] / k).mean())
    rec = float((hit[ok_r] / n_rel[ok_r]).mean())
    return prec, rec


# ------------------------------------------------------------------ CBF
class GenreCBF:
    """Профиль пользователя p_u = sum_j (r_uj - mu - b_u) x_j,
    прогноз: mu + b_u + lam * cos(p_u, x_i)."""

    def __init__(self, X, m):
        self.X, self.m = X, m
        self.xn = np.linalg.norm(X, axis=1) + 1e-12

    def fit(self, train):
        self.mu = float(train["r"].mean())
        self.bu, self.cnt = user_bias(train, self.m, self.mu)
        resid = train["r"].values - self.mu - self.bu[train["u"].values]
        P = np.zeros((self.m, self.X.shape[1]))
        np.add.at(P, train["u"].values, resid[:, None] * self.X[train["i"].values])
        pn = np.linalg.norm(P, axis=1) + 1e-12
        self.C = (P @ self.X.T) / (pn[:, None] * self.xn[None, :])
        return self

    def predict(self, df, lam):
        u, i = df["u"].values, df["i"].values
        base = self.mu + np.where(self.cnt[u] > 0, self.bu[u], 0.0)
        return np.clip(base + lam * self.C[u, i], 1, 5)


# ------------------------------------------------------------------ один прогон
def run_one(seed, q, train_full, test, X, m, rng_rank):
    n_keep = int(np.ceil(q * len(train_full)))
    tr = train_full.iloc[:n_keep]          # вложенные подмножества (train_full уже перемешан)

    # --- валидация для подбора гиперпараметров
    perm = np.random.RandomState(1000 + seed).permutation(len(tr))
    n_val = max(int(0.2 * len(tr)), 1)
    val = tr.iloc[perm[:n_val]]
    tr_in = tr.iloc[perm[n_val:]]
    mu_in = float(tr_in["r"].mean())
    bu_in, cnt_in = user_bias(tr_in, m, mu_in)

    best = {}
    ts_in = to_trainset(tr_in)
    for name, grid in [("UserKNN", GRID_KNN), ("ItemKNN", GRID_KNN), ("SVD", GRID_SVD)]:
        scores = []
        for g in grid:
            if name == "SVD":
                algo = SVD(n_factors=g[0], reg_all=g[1], n_epochs=20, lr_all=0.005, random_state=seed)
            else:
                algo = KNNWithMeans(k=g, sim_options={"name": "msd", "user_based": name == "UserKNN"},
                                    verbose=False)
            algo.fit(ts_in)
            p = surprise_predict(algo, val, bu_in, cnt_in, mu_in)
            scores.append(rmse_mae(val["r"].values, p)[0])
        j = int(np.argmin(scores))
        best[name] = (grid[j], scores[j])
    cbf_in = GenreCBF(X, m).fit(tr_in)
    sc = [rmse_mae(val["r"].values, cbf_in.predict(val, lam))[0] for lam in GRID_LAMBDA]
    best["CBF"] = (GRID_LAMBDA[int(np.argmin(sc))], float(min(sc)))

    # --- итоговое обучение на всей Omega^(q) и тест
    mu = float(tr["r"].mean())
    bu, cnt = user_bias(tr, m, mu)
    ts = to_trainset(tr)
    y = test["r"].values
    known_u = float(np.mean(cnt[test["u"].values] > 0))
    item_cnt = np.bincount(tr["i"].values, minlength=X.shape[0])
    known_i = float(np.mean(item_cnt[test["i"].values] > 0))
    both = float(np.mean((cnt[test["u"].values] > 0) & (item_cnt[test["i"].values] > 0)))

    preds = {}
    for name in ["UserKNN", "ItemKNN", "SVD"]:
        g = best[name][0]
        if name == "SVD":
            algo = SVD(n_factors=g[0], reg_all=g[1], n_epochs=20, lr_all=0.005, random_state=seed)
        else:
            algo = KNNWithMeans(k=g, sim_options={"name": "msd", "user_based": name == "UserKNN"},
                                verbose=False)
        algo.fit(ts)
        preds[name] = surprise_predict(algo, test, bu, cnt, mu)
    algo = BaselineOnly(verbose=False)
    algo.fit(ts)
    preds["Bias-UI"] = surprise_predict(algo, test, bu, cnt, mu)
    preds["Bias-U"] = np.clip(fallback_pred(test["u"].values, bu, cnt, mu), 1, 5)
    cbf = GenreCBF(X, m).fit(tr)
    preds["CBF"] = cbf.predict(test, best["CBF"][0])

    # лучший CF по валидационной RMSE
    cf_name = min(["UserKNN", "ItemKNN", "SVD"], key=lambda n: best[n][1])
    preds["CF"] = preds[cf_name]

    rows = []
    rho = len(tr) / (m * X.shape[0])
    for name, p in preds.items():
        rm, ma = rmse_mae(y, p)
        pr, re = rank_metrics(test, p, rng_rank)
        rows.append(dict(seed=seed, q=q, rho=rho, n_train=len(tr), model=name,
                         rmse=rm, mae=ma, p10=pr, r10=re,
                         known_user=known_u, known_item=known_i, known_both=both,
                         cf_choice=cf_name if name == "CF" else "",
                         params=str(best.get(name, best.get(cf_name) if name == "CF" else ""))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--data", default=".", help="папка с u.data/u.item или ml.inter/ml.item")
    args = ap.parse_args()

    ratings, X, genres = load_data(args.data)
    m = ratings["u"].nunique()
    print(f"m={m}, n={X.shape[0]}, |Omega|={len(ratings)}, genres={len(genres)}", flush=True)

    all_rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        rs = np.random.RandomState(seed)
        perm = rs.permutation(len(ratings))
        n_test = int(0.2 * len(ratings))
        test = ratings.iloc[perm[:n_test]].reset_index(drop=True)
        train_full = ratings.iloc[perm[n_test:]].reset_index(drop=True)   # уже в случайном порядке
        rng_rank = np.random.RandomState(10_000 + seed)
        for q in Q_LIST:
            all_rows += run_one(seed, q, train_full, test, X, m, rng_rank)
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
