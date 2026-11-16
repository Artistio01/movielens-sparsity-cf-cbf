"""
Рисунки для статьи (векторные PDF, шрифты встраиваются как Type 42).
Запуск:  python figures.py [ru|en]
Нужны: results.csv (после experiment.py), ml.inter, ml.item.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

LANG = sys.argv[1] if len(sys.argv) > 1 else "ru"
OUT = "images" if LANG == "ru" else "images_en"
os.makedirs(OUT, exist_ok=True)

T = {
    "ru": dict(
        rho="Плотность $\\rho$, % (→ разреженнее)", sparse="",
        rmse="RMSE на тесте", p10="Precision@10 на тесте",
        delta="$\\Delta = \\mathrm{RMSE}(\\mathrm{CF}) - \\mathrm{RMSE}(\\mathrm{CBF})$",
        cbf_better="CBF лучше", cf_better="CF лучше",
        share="Доля тестовых пар, %", known_item="фильм есть в обучении", known_both="пользователь и фильм есть",
        ratings="Оценка", n_ratings="Число оценок, тыс.", genres="Число фильмов", base="случайный порядок",
        panel_a="а)", panel_b="б)", users="Число оценок у пользователя", n_users="Число пользователей",
        full="полная матрица", dens="плотность",
    ),
    "en": dict(
        rho="Density $\\rho$, % (→ sparser)", sparse="",
        rmse="Test RMSE", p10="Test Precision@10",
        delta="$\\Delta = \\mathrm{RMSE}(\\mathrm{CF}) - \\mathrm{RMSE}(\\mathrm{CBF})$",
        cbf_better="CBF better", cf_better="CF better",
        share="Share of test pairs, %", known_item="movie seen in training", known_both="user and movie seen",
        ratings="Rating", n_ratings="Ratings, thousands", genres="Number of movies", base="random order",
        panel_a="a)", panel_b="b)", users="Ratings per user", n_users="Number of users",
        full="full matrix", dens="density",
    ),
}[LANG]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
})

C = dict(CBF="#D55E00", SVD="#0072B2", UserKNN="#009E73", ItemKNN="#CC79A7",
         **{"Bias-U": "#7f7f7f", "Bias-UI": "#7f7f7f"})
LS = {"Bias-U": (0, (4, 2)), "Bias-UI": (0, (1, 1.5))}
LABEL = {"CBF": "CBF (жанры)" if LANG == "ru" else "CBF (genres)", "SVD": "SVD", "UserKNN": "User-kNN",
         "ItemKNN": "Item-kNN", "Bias-U": "Bias-U", "Bias-UI": "Bias-UI"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=170)
    plt.close(fig)


res = pd.read_csv("results.csv")
res["rho_pct"] = res["rho"] * 100
S = res["seed"].nunique()


def agg(model, col):
    d = res[res.model == model].groupby("q").agg(rho=("rho_pct", "mean"), m=(col, "mean"), s=(col, "std"))
    return d.sort_values("rho")


# ------------------------------------------------------------ Рис. 1: схема эксперимента
def fig_pipeline():
    ru = LANG == "ru"
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=7.6, weight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.8, body, ha="center", va="top", fontsize=6.4, linespacing=1.3)

    def arrow(p, q_):
        ax.add_patch(FancyArrowPatch(p, q_, arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#444"))

    W, H = 3.3, 2.55
    X0, X1, X2 = 0.05, 3.85, 7.65
    YT, YB = 3.75, 0.1
    box(X0, YT, W, H, "1  Данные" if ru else "1  Data",
        "MovieLens 100K\n943 пользователя\n1682 фильма\n100 000 оценок\n19 жанров" if ru else
        "MovieLens 100K\n943 users\n1682 movies\n100,000 ratings\n19 genres", "#EAF3FB", "#0072B2")
    box(X1, YT, W, H, "2  Разбиение" if ru else "2  Split",
        "случайное:\n80 % обучение\n20 % тест\n(тест не меняется)" if ru else
        "random:\n80% train\n20% test\n(test is fixed)", "#EAF3FB", "#0072B2")
    box(X2, YT, W, H, "3  Разрежение" if ru else "3  Sparsification",
        "оставляем долю $q$\nобучающих оценок:\n$q$ = 1 … 0,02\n$\\rho$ = 5 % … 0,1 %" if ru else
        "keep share $q$ of\ntraining ratings:\n$q$ = 1 … 0.02\n$\\rho$ = 5% … 0.1%", "#FDF1E1", "#B56B00")
    box(X0, YB, W, H, "4  Модели" if ru else "4  Models",
        "CF: User-kNN,\nItem-kNN, SVD\nCBF: жанры фильмов\nориентиры: Bias-U/UI\nподбор на валидации" if ru else
        "CF: User-kNN,\nItem-kNN, SVD\nCBF: movie genres\nreferences: Bias-U/UI\ntuned on validation",
        "#E8F6F0", "#009E73")
    box(X1, YB, W, H, "5  Оценка" if ru else "5  Evaluation",
        "на тесте:\nRMSE, MAE,\nPrecision@10,\nRecall@10\n10 повторов" if ru else
        "on test:\nRMSE, MAE,\nPrecision@10,\nRecall@10\n10 repeats", "#F6EAF1", "#A64F82")
    box(X2, YB, W, H, "6  Сравнение" if ru else "6  Comparison",
        "разность $\\Delta(q)$\nпарный тест\nУилкоксона\nпорог $\\rho^{*}$" if ru else
        "difference $\\Delta(q)$\npaired Wilcoxon\ntest\nthreshold $\\rho^{*}$", "#F6EAF1", "#A64F82")

    arrow((X0 + W + 0.05, YT + H / 2), (X1 - 0.05, YT + H / 2))
    arrow((X1 + W + 0.05, YT + H / 2), (X2 - 0.05, YT + H / 2))
    # ломаная стрелка от блока 3 к блоку 4
    ym = (YT + YB + H) / 2
    xs = X2 + W / 2
    ax.plot([xs, xs], [YT - 0.05, ym], color="#444", lw=1.1)
    ax.plot([xs, X0 + W / 2], [ym, ym], color="#444", lw=1.1)
    arrow((X0 + W / 2, ym), (X0 + W / 2, YB + H + 0.05))
    arrow((X0 + W + 0.05, YB + H / 2), (X1 - 0.05, YB + H / 2))
    arrow((X1 + W + 0.05, YB + H / 2), (X2 - 0.05, YB + H / 2))
    save(fig, "fig_pipeline")


# ------------------------------------------------------------ Рис. 2: данные
def fig_data():
    r = pd.read_csv("ml.inter", sep="\t"); r.columns = ["u", "i", "r", "t"]
    it = pd.read_csv("ml.item", sep="\t"); it.columns = ["i", "title", "year", "cls"]
    cnt = {}
    for c in it["cls"].fillna("unknown"):
        for g in str(c).split():
            cnt[g] = cnt.get(g, 0) + 1
    g = pd.Series(cnt).sort_values()
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.15), gridspec_kw=dict(width_ratios=[0.8, 1.0, 1.15], wspace=0.55))
    a = axs[0]
    vc = r["r"].value_counts().sort_index()
    a.bar(vc.index, vc.values / 1000, color="#0072B2", width=0.7)
    a.set_xlabel(T["ratings"]); a.set_ylabel(T["n_ratings"]); a.set_xticks([1, 2, 3, 4, 5])
    a.set_title(T["panel_a"], loc="left", weight="bold")
    a = axs[1]
    pu = r.groupby("u").size()
    a.hist(pu, bins=np.logspace(np.log10(15), np.log10(750), 22), color="#009E73")
    a.set_xscale("log"); a.set_xticks([20, 50, 100, 200, 500]); a.set_xticklabels(["20", "50", "100", "200", "500"]); a.minorticks_off()
    a.set_xlabel(T["users"]); a.set_ylabel(T["n_users"])
    a.set_title(T["panel_b"], loc="left", weight="bold")
    a = axs[2]
    a.barh(g.index, g.values, color="#D55E00", height=0.7)
    a.tick_params(axis="y", labelsize=6.2)
    a.set_xlabel(T["genres"]); a.set_title("в)" if LANG == "ru" else "c)", loc="left", weight="bold")
    save(fig, "fig_data")


# ------------------------------------------------------------ Рис. 3: наглядная разреженность
def fig_sparse():
    r = pd.read_csv("ml.inter", sep="\t"); r.columns = ["u", "i", "r", "t"]
    r["u"] = r["u"].rank(method="dense").astype(int) - 1
    r["i"] = r["i"].rank(method="dense").astype(int) - 1
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(r))
    tr = r.iloc[perm[20000:]].reset_index(drop=True)          # 80 % обучения
    users = rs.choice(943, 90, replace=False)
    items = rs.choice(1682, 150, replace=False)
    users = users[np.argsort(-r.groupby("u").size().reindex(users).values)]     # по убыванию активности
    items = items[np.argsort(-r.groupby("i").size().reindex(items).fillna(0).values)]  # по убыванию популярности
    umap = {u: k for k, u in enumerate(users)}; imap = {i: k for k, i in enumerate(items)}
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.95), gridspec_kw=dict(wspace=0.08))
    for ax, q in zip(axs, [1.0, 0.25, 0.05]):
        sub = tr.iloc[: int(np.ceil(q * len(tr)))]
        sub = sub[sub["u"].isin(umap) & sub["i"].isin(imap)]
        x = sub["i"].map(imap).values; y = sub["u"].map(umap).values
        ax.scatter(x, y, s=3.2, c="#0072B2", marker="s", linewidths=0)
        ax.set_xlim(-1, len(items)); ax.set_ylim(len(users), -1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#888"); s.set_linewidth(0.6)
        rho = q * len(tr) / (943 * 1682) * 100
        ax.set_title(f"$q$ = {str(q).replace('.', ',') if LANG == 'ru' else q}, $\\rho$ = {rho:.2f}".replace(".", ",") + " %" if LANG == "ru"
                     else f"$q$ = {q}, $\\rho$ = {rho:.2f}%", fontsize=7.6)
    axs[0].set_ylabel("пользователи" if LANG == "ru" else "users", fontsize=7.5)
    for ax in axs:
        ax.set_xlabel("фильмы" if LANG == "ru" else "movies", fontsize=7.5)
    save(fig, "fig_sparse")


# ------------------------------------------------------------ Рис. 4: RMSE и Precision@10
def fig_main():
    models = ["Bias-U", "CBF", "UserKNN", "ItemKNN", "SVD"]
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.45), gridspec_kw=dict(wspace=0.42))
    for ax, col, yl, pn in [(axs[0], "rmse", T["rmse"], T["panel_a"]), (axs[1], "p10", T["p10"], T["panel_b"])]:
        for mname in models:
            d = agg(mname, col)
            ax.plot(d["rho"], d["m"], color=C[mname], lw=1.5, ls=LS.get(mname, "-"),
                    marker="o" if mname not in LS else None, ms=2.8, label=LABEL[mname])
            ax.fill_between(d["rho"], d["m"] - d["s"], d["m"] + d["s"], color=C[mname], alpha=0.14, lw=0)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xticks([5, 2, 1, 0.5, 0.2, 0.1]); ax.set_xticklabels(["5", "2", "1", "0,5", "0,2", "0,1"] if LANG == "ru" else ["5", "2", "1", "0.5", "0.2", "0.1"])
        ax.minorticks_off()
        ax.set_xlabel(T["rho"]); ax.set_ylabel(yl); ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(pn, loc="left", weight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.13), columnspacing=1.2, handlelength=1.8)
    save(fig, "fig_main")


# ------------------------------------------------------------ Рис. 5: Delta и покрытие
def fig_delta():
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.45), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    cbf = res[res.model == "CBF"].set_index(["seed", "q"])
    for mname in ["UserKNN", "ItemKNN", "SVD"]:
        cf = res[res.model == mname].set_index(["seed", "q"])
        d = (cf["rmse"] - cbf["rmse"]).reset_index(name="d")
        d["rho"] = d["q"].map(res.groupby("q")["rho_pct"].mean())
        g = d.groupby("q").agg(rho=("rho", "mean"), m=("d", "mean"), s=("d", "std")).sort_values("rho")
        ax.errorbar(g["rho"], g["m"], yerr=g["s"] / np.sqrt(S) * 1.96, color=C[mname], lw=1.5, marker="o", ms=2.8,
                    capsize=1.8, elinewidth=0.8, label=LABEL[mname])
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
    ax.set_xticks([5, 2, 1, 0.5, 0.2, 0.1]); ax.set_xticklabels(["5", "2", "1", "0,5", "0,2", "0,1"] if LANG == "ru" else ["5", "2", "1", "0.5", "0.2", "0.1"])
    ax.set_xlabel(T["rho"]); ax.set_ylabel(T["delta"]); ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-0.14, 0.16)
    ax.text(0.03, 0.97, T["cbf_better"] + " ↑", transform=ax.transAxes, va="top", fontsize=6.8, color="#D55E00")
    ax.text(0.03, 0.03, T["cf_better"] + " ↓", transform=ax.transAxes, va="bottom", fontsize=6.8, color="#0072B2")
    ax.legend(loc="center left", ncol=1, bbox_to_anchor=(0.0, 0.68))
    ax.set_title(T["panel_a"], loc="left", weight="bold")
    ax = axs[1]
    k = res[res.model == "SVD"].groupby("q").agg(rho=("rho_pct", "mean"), ki=("known_item", "mean"),
                                                 kb=("known_both", "mean"), ku=("known_user", "mean")).sort_values("rho")
    ax.plot(k["rho"], k["ki"] * 100, color="#56B4E9", lw=1.5, marker="o", ms=2.8, label=T["known_item"])
    ax.plot(k["rho"], k["kb"] * 100, color="#0072B2", lw=1.5, marker="o", ms=2.8, label=T["known_both"])
    ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
    ax.set_xticks([5, 2, 1, 0.5, 0.2, 0.1]); ax.set_xticklabels(["5", "2", "1", "0,5", "0,2", "0,1"] if LANG == "ru" else ["5", "2", "1", "0.5", "0.2", "0.1"])
    ax.set_xlabel(T["rho"]); ax.set_ylabel(T["share"]); ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(55, 101); ax.legend(loc="lower left")
    ax.set_title(T["panel_b"], loc="left", weight="bold")
    save(fig, "fig_delta")


if __name__ == "__main__":
    fig_pipeline(); fig_data(); fig_sparse(); fig_main(); fig_delta()
    print("ok", OUT)
