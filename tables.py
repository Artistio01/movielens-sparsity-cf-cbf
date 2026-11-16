"""Таблицы LaTeX и статистика из results.csv.  Запуск: python tables.py"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv("results.csv")
d["rho_pct"] = d["rho"] * 100
S = d["seed"].nunique()
QS = [1.0, 0.25, 0.15, 0.1, 0.05, 0.02]
ROWS = [("Bias-U", "Bias-U (справочно)"), ("Bias-UI", "Bias-UI (справочно)"),
        ("CBF", "CBF (жанры)"), ("UserKNN", "User-kNN"), ("ItemKNN", "Item-kNN"), ("SVD", "SVD")]
MAIN = ["CBF", "UserKNN", "ItemKNN", "SVD"]


def num(x, nd=3):
    return f"{x:.{nd}f}".replace(".", "{,}")


def make_table(col, caption, label, higher_better):
    g = d.groupby(["q", "model"])[col].agg(["mean", "std"]).reset_index()
    rho = d.groupby("q")["rho_pct"].mean()
    lines = [r"\begin{table}[t]", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}", r"  \centering", r"  \footnotesize",
             r"  \setlength{\tabcolsep}{3.2pt}", r"  \begin{tabular}{l" + "c" * len(QS) + "}", r"    \toprule",
             "    $q$ & " + " & ".join(num(q, 2) for q in QS) + r" \\",
             r"    $\rho$, \% & " + " & ".join(num(rho[q], 2) for q in QS) + r" \\", r"    \midrule"]
    best = {}
    for q in QS:
        sub = g[(g.q == q) & (g.model.isin(MAIN))]
        best[q] = sub.loc[sub["mean"].idxmax() if higher_better else sub["mean"].idxmin(), "model"]
    for key, name in ROWS:
        if key == "CBF":
            lines.append(r"    \midrule")
        cells = []
        for q in QS:
            r = g[(g.q == q) & (g.model == key)].iloc[0]
            cell = f"{num(r['mean'])}$\\pm${num(r['std'])}"
            if key in MAIN and best[q] == key:
                cell = r"\textbf{" + f"{num(r['mean'])}" + r"}$\pm$" + num(r["std"])
            cells.append(cell)
        lines.append(f"    {name} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


t1 = make_table("rmse", r"RMSE на тесте (среднее $\pm$ стд. отклонение по 10 повторам); меньше --- лучше. "
                r"Жирным выделен лучший из четырёх основных методов в каждом столбце.", "tab:rmse", False)
t2 = make_table("p10", r"Precision@10 на тесте (среднее $\pm$ стд. отклонение по 10 повторам); больше --- лучше. "
                r"Bias-U выдаёт для пользователя одинаковые прогнозы, поэтому порядок случайный.", "tab:p10", True)
open("tab_rmse.tex", "w", encoding="utf-8").write(t1)
open("tab_p10.tex", "w", encoding="utf-8").write(t2)

# ------------------------------------------------------ статистика для текста
pd.set_option("display.width", 220)
cbf = d[d.model == "CBF"].set_index(["seed", "q"])
print("rho by q:", d.groupby("q")["rho_pct"].mean().round(3).to_dict())
for metric, sign in [("rmse", 1), ("p10", -1)]:
    print("\n== Delta", metric, "(positive = CBF better)")
    for a in ["UserKNN", "ItemKNN", "SVD", "CF", "Bias-UI", "Bias-U"]:
        x = d[d.model == a].set_index(["seed", "q"])
        dl = ((x[metric] - cbf[metric]) * sign).reset_index(name="dl")
        row = []
        for q, s in dl.groupby("q"):
            v = s.dl.values
            try:
                p = wilcoxon(v).pvalue
            except Exception:
                p = np.nan
            row.append((q, round(v.mean(), 3), int((v > 0).sum()), round(p, 4)))
        print(a, sorted(row, reverse=True))
print("\nCF choice:")
print(d[d.model == "CF"].groupby(["q", "cf_choice"]).size().unstack(fill_value=0))
print("\ncoverage (SVD rows):")
print(d[d.model == "SVD"].groupby("q")[["known_user", "known_item", "known_both"]].mean().round(3).sort_index(ascending=False))
print("\nmean values")
for col in ["rmse", "mae", "p10", "r10"]:
    print(col)
    print(d.groupby(["q", "model"])[col].mean().unstack().round(3).sort_index(ascending=False)[["Bias-U", "Bias-UI", "CBF", "UserKNN", "ItemKNN", "SVD", "CF"]])
print("\nchosen k / lambda:")
import ast
for m in ["UserKNN", "ItemKNN", "CBF"]:
    x = d[d.model == m].copy()
    x["hp"] = x.params.apply(lambda s: str(ast.literal_eval(s)[0]))
    print(m)
    print(x.groupby(["q", "hp"]).size().unstack(fill_value=0).sort_index(ascending=False))
