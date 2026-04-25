# # =============================================================
# #  AIF360 Bias Detection & Mitigation Pipeline  v4
# #  Changes from v3:
# #    - plot_results: figure renders INLINE in the notebook cell
# #      (no separate saved file, no plt.show() popup)
# #    - Radar chart: explicit ylim set so all stage lines are
# #      visible — Reweighing line was crushed to centre before
# #    - Radar chart: grid lines + radial ticks added for clarity
# #    - ax4 x-tick labels fixed (were misaligned after rotation)
# # =============================================================

# # !pip install "aif360[all]" numpy pandas //TE commented prev code

# import io
# import pandas as pd
# import numpy as np
# import hashlib
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec
# import warnings
# warnings.filterwarnings("ignore")

# # ── Inline display helper ─────────────────────────────────────
# # Works in Jupyter / Colab / VS Code notebooks.
# # Falls back to plt.show() when running as a plain script.
# # try: //TE commented prev code start
# #     from IPython.display import display, Image as IPImage
# #     import IPython
# #     _IN_NOTEBOOK = True
# # except ImportError:
# #     _IN_NOTEBOOK = False //TE commented prev code end

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score
# )

# from aif360.datasets import BinaryLabelDataset
# from aif360.metrics import ClassificationMetric
# from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
# from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
# from aif360.algorithms.inprocessing import AdversarialDebiasing

# # import tensorflow as tf //TE for tensor error commented prev code

# try:
#     from imblearn.over_sampling import SMOTE
#     SMOTE_AVAILABLE = True
# except ImportError:
#     SMOTE_AVAILABLE = False
#     print("⚠️  imbalanced-learn not found — SMOTE step will be skipped.")
#     print("   Install with:  pip install imbalanced-learn\n")


# # =============================================================
# # THRESHOLDS
# # =============================================================
# BIAS_THRESHOLD = 0.10   # SPD / EOD / AOD
# DI_THRESHOLD   = 0.20   # |1 - DI|  (looser — DI is harder to fix)
# DI_WEIGHT      = 1.5    # extra penalty for DI in composite score


# # =============================================================
# # HELPERS
# # =============================================================
# def safe(v):
#     if v is None:
#         return 0.0
#     try:
#         f = float(v)
#         return 0.0 if np.isnan(f) else f
#     except (TypeError, ValueError):
#         return 0.0


# def get_metric_dict(cm, y_true, y_pred):
#     return {
#         "SPD":       safe(cm.statistical_parity_difference()),
#         "DI":        safe(cm.disparate_impact()),
#         "EOD":       safe(cm.equal_opportunity_difference()),
#         "AOD":       safe(cm.average_odds_difference()),
#         "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
#         "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
#         "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
#         "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
#     }


# def fairness_score(metrics):
#     """DI-weighted composite — lower = fairer."""
#     return (
#         abs(metrics["SPD"]) +
#         DI_WEIGHT * abs(1 - metrics["DI"]) +
#         abs(metrics["EOD"]) +
#         abs(metrics["AOD"])
#     )


# def is_biased(metrics):
#     return (
#         abs(metrics["SPD"])     > BIAS_THRESHOLD or
#         abs(metrics["EOD"])     > BIAS_THRESHOLD or
#         abs(metrics["AOD"])     > BIAS_THRESHOLD or
#         abs(1 - metrics["DI"]) > DI_THRESHOLD
#     )


# def print_report(metrics, stage):
#     print(f"\n{'='*55}")
#     print(f"  {stage}")
#     print(f"{'='*55}")
#     print(f"  FAIRNESS METRICS")
#     print(f"    SPD  : {metrics['SPD']:+.4f}   (ideal  0)")
#     print(f"    DI   :  {metrics['DI']:.4f}   (ideal  1 | |1-DI|={abs(1-metrics['DI']):.4f})")
#     print(f"    EOD  : {metrics['EOD']:+.4f}   (ideal  0)")
#     print(f"    AOD  : {metrics['AOD']:+.4f}   (ideal  0)")
#     print(f"  COMPOSITE (DI-weighted) : {fairness_score(metrics):.4f}")
#     print(f"  PERFORMANCE METRICS")
#     print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
#     print(f"    F1        : {metrics['F1']:.4f}")
#     print(f"    Precision : {metrics['Precision']:.4f}")
#     print(f"    Recall    : {metrics['Recall']:.4f}")
#     flag = "⚠️  BIAS DETECTED" if is_biased(metrics) else "✅  WITHIN THRESHOLD"
#     print(f"\n  {flag}")


# # =============================================================
# # DATA HELPERS
# # =============================================================
# def show_columns(df):
#     print(f"\n📊 DATASET SHAPE: {df.shape}")
#     print("\n📌 COLUMNS:")
#     for col in df.columns:
#         print(f"   - {col}")
#     print("\n📌 SAMPLE:")
#     print(df.head())


# def get_user_columns(df):
#     while True:
#         target = input("\nEnter TARGET column: ").strip()
#         if target not in df.columns:
#             print("❌ Invalid TARGET"); continue
#         print("  Unique values:", df[target].unique())

#         protected = input("\nEnter PROTECTED column: ").strip()
#         if protected not in df.columns:
#             print("❌ Invalid PROTECTED"); continue
#         print("  Unique values:", df[protected].unique())
#         return target, protected


# def diagnose_group_rates(df, target, protected):
#     print("\n" + "─"*55)
#     print("  GROUP POSITIVE-RATE DIAGNOSIS")
#     print("─"*55)
#     rates = {}
#     for g in sorted(df[protected].unique()):
#         mask  = df[protected] == g
#         rate  = df.loc[mask, target].mean()
#         label = "PRIVILEGED" if g == 1 else "UNPRIVILEGED"
#         print(f"  Group {g} ({label:12s}):  P(y=1) = {rate:.4f}  (n={mask.sum():,})")
#         rates[g] = rate
#     if 0 in rates and 1 in rates and rates[1] > 0:
#         ratio = rates[0] / rates[1]
#         print(f"\n  Base-rate ratio (unpriv ÷ priv) = {ratio:.4f}")
#         if ratio < 0.1:
#             print("  ⚠️  SEVERE imbalance — DI needs dataset-level fix.")
#         elif ratio < 0.5:
#             print("  ⚠️  Moderate imbalance — SMOTE / DIR strongly recommended.")
#         else:
#             print("  ✅  Reasonable balance — standard mitigation should work.")
#     return rates


# def feature_engineering(df):
#     df = df.copy()
#     for col in list(df.columns):
#         if "time" in col.lower():
#             df[col] = pd.to_datetime(df[col], errors="coerce")
#             df[col + "_hour"] = df[col].dt.hour
#             df.drop(columns=[col], inplace=True)
#         elif "ip" in col.lower():
#             df[col] = df[col].astype(str).apply(
#                 lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 1000
#             )
#     return df


# def clean_data(df, target, protected):
#     df = df.copy()
#     df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

#     cat_cols = df.select_dtypes(include=["object"]).columns
#     safe_cols = []
#     high_card = []

#     for col in cat_cols:
#         if col == target:
#             continue
#         elif col == protected:
#             df[col] = LabelEncoder().fit_transform(df[col].astype(str))
#         elif df[col].nunique() <= 20:
#             safe_cols.append(col)
#         else:
#             high_card.append(col)

#     # Drop high-cardinality columns
#     df = df.drop(columns=high_card)

#     # Label encode safe categorical columns (instead of one-hot)
#     for col in safe_cols:
#         df[col] = LabelEncoder().fit_transform(df[col].astype(str))

#     # Reduce memory usage
#     df = df.astype("float32")
#     return df


# def preprocess_labels(df, target, protected, fav, priv): #TE added extra , fav, priv
#     df = df.copy()
#     df[target]    = df[target].astype(str).str.lower()
#     df[protected] = df[protected].astype(str).str.lower()

#     print("\nTarget values:", df[target].unique())
#     # fav = input("Enter FAVORABLE label: ").lower() //TE commented prev code
#     # df[target] = df[target].apply(lambda x: 1 if x == fav else 0) //TE commented prev code and replace with following code
#     df[target] = df[target].apply( #TE start
#     lambda x: 1 if str(x).lower() == str(fav).lower() else 0 #TE end
# )

#     print("\nProtected values:", df[protected].unique())
#     # priv = input("Enter PRIVILEGED group: ").lower() //TE commented prev code
#     # df[protected] = df[protected].apply(lambda x: 1 if x == priv else 0) //TE commented prev code to replace with following code
#     df[protected] = df[protected].apply( #TE start
#     lambda x: 1 if str(x).lower() == str(priv).lower() else 0  #TE end
# )
#     return df


# def make_dataset(df, target, protected):
#     return BinaryLabelDataset(
#         df=df,
#         label_names=[target],
#         protected_attribute_names=[protected]
#     )


# def make_ds(X_arr, y_arr, p_arr, target, protected):
#     df_tmp = pd.DataFrame(X_arr)
#     df_tmp[target]    = np.array(y_arr)
#     df_tmp[protected] = np.array(p_arr)
#     return make_dataset(df_tmp, target, protected)


# # =============================================================
# # INLINE DISPLAY UTILITY
# # =============================================================
# # def _show_inline(fig): //TE commented prev code start
# #     """
# #     Render figure bytes directly into the notebook cell.
# #     No file is written to disk. Falls back to plt.show() in
# #     plain-script mode.
# #     """
# #     if _IN_NOTEBOOK:
# #         buf = io.BytesIO()
# #         fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
# #                     facecolor=fig.get_facecolor())
# #         buf.seek(0)
# #         display(IPImage(data=buf.read()))
# #         plt.close(fig)
# #     else:
# #         plt.tight_layout()
# #         plt.show()  //TE commented prev code end


# # =============================================================
# # 6-PANEL PLOT  — inline + radar bug fixed
# # =============================================================
# def plot_results(results):
#     """
#     Panel 1 — Fairness deviation bars
#     Panel 2 — Performance bars
#     Panel 3 — Radar chart  (ylim now explicit — all lines visible)
#     Panel 4 — DI-weighted composite score
#     Panel 5 — DI raw ratio line chart
#     Panel 6 — SPD vs |1-DI| trade-off scatter
#     """
#     stages = list(results.keys())
#     n      = len(stages)

#     # colour palette — one distinct colour per stage
#     palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2",
#                "#937860", "#DA8BC3", "#64B5CD", "#CCB974"]
#     colors  = palette[:n]

#     # ── collect values ────────────────────────────────────────
#     spd  = [abs(results[s]["SPD"])     for s in stages]
#     di_d = [abs(1 - results[s]["DI"])  for s in stages]
#     di_r = [results[s]["DI"]           for s in stages]
#     eod  = [abs(results[s]["EOD"])     for s in stages]
#     aod  = [abs(results[s]["AOD"])     for s in stages]
#     acc  = [results[s]["Accuracy"]     for s in stages]
#     f1   = [results[s]["F1"]           for s in stages]
#     prec = [results[s]["Precision"]    for s in stages]
#     rec  = [results[s]["Recall"]       for s in stages]
#     comp = [fairness_score(results[s]) for s in stages]

#     x     = np.arange(n)
#     bar_w = 0.18

#     fig = plt.figure(figsize=(20, 19))
#     fig.patch.set_facecolor("#F8F9FA")
#     gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

#     # ── Panel 1: Fairness deviation bars ──────────────────────
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.set_facecolor("#FFFFFF")
#     m_labels = ["SPD", "|1−DI|", "EOD", "AOD"]
#     m_vals   = [spd, di_d, eod, aod]
#     m_colors = ["#4C72B0", "#E24B4A", "#55A868", "#C44E52"]
#     for i, (vals, mc, ml) in enumerate(zip(m_vals, m_colors, m_labels)):
#         bars = ax1.bar(x + i * bar_w, vals, bar_w, label=ml,
#                        color=mc, alpha=0.85, edgecolor="white", linewidth=0.5)
#         for bar, v in zip(bars, vals):
#             ax1.text(bar.get_x() + bar.get_width() / 2,
#                      bar.get_height() + 0.004,
#                      f"{v:.3f}", ha="center", va="bottom",
#                      fontsize=7, color="#222222")
#     ax1.axhline(BIAS_THRESHOLD, color="red", linestyle="--",
#                 linewidth=1.2, label=f"Threshold ({BIAS_THRESHOLD})")
#     ax1.set_title("Fairness metrics — deviation from ideal",
#                   fontsize=11, fontweight="bold", pad=8)
#     ax1.set_xticks(x + bar_w * 1.5)
#     ax1.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
#     ax1.set_ylabel("Absolute deviation (lower = better)")
#     ax1.legend(fontsize=8)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ymax = max(max(spd), max(di_d), max(eod), max(aod))
#     ax1.set_ylim(0, ymax * 1.35 + 0.02)

#     # ── Panel 2: Performance bars ──────────────────────────────
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.set_facecolor("#FFFFFF")
#     p_labels = ["Accuracy", "F1", "Precision", "Recall"]
#     p_vals   = [acc, f1, prec, rec]
#     p_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
#     for i, (vals, pc, pl) in enumerate(zip(p_vals, p_colors, p_labels)):
#         bars = ax2.bar(x + i * bar_w, vals, bar_w, label=pl,
#                        color=pc, alpha=0.85, edgecolor="white", linewidth=0.5)
#         for bar, v in zip(bars, vals):
#             ax2.text(bar.get_x() + bar.get_width() / 2,
#                      bar.get_height() + 0.004,
#                      f"{v:.3f}", ha="center", va="bottom",
#                      fontsize=7, color="#222222")
#     ax2.set_title("Performance metrics (higher = better)",
#                   fontsize=11, fontweight="bold", pad=8)
#     ax2.set_xticks(x + bar_w * 1.5)
#     ax2.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
#     ax2.set_ylabel("Score")
#     ax2.set_ylim(0, 1.15)
#     ax2.legend(fontsize=8, loc="lower right")
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)

#     # ── Panel 3: Radar chart — BUG FIXED ──────────────────────
#     # Root cause of missing Reweighing line:
#     #   ax3.set_ylim() was never called, so matplotlib auto-scaled
#     #   to the maximum values. Stages with large EOD/AOD dominated
#     #   the scale, squashing smaller-valued stages (like Reweighing
#     #   after mitigation) into an invisible ring near the origin.
#     # Fix: compute a common upper limit with 20% headroom and set
#     #   it explicitly, then add clear grid rings so every line
#     #   is visible regardless of scale differences.
#     ax3 = fig.add_subplot(gs[1, 0], polar=True)
#     ax3.set_facecolor("#FFFFFF")

#     radar_labels = ["SPD", "|1−DI|", "EOD", "AOD", "1−F1"]
#     num_v  = len(radar_labels)
#     angles = np.linspace(0, 2 * np.pi, num_v, endpoint=False).tolist()
#     angles += angles[:1]   # close the polygon

#     # Compute all radar values first so we know the max
#     radar_data = {}
#     for stage in stages:
#         r = results[stage]
#         v = [abs(r["SPD"]), abs(1 - r["DI"]),
#              abs(r["EOD"]), abs(r["AOD"]), 1 - r["F1"]]
#         v += v[:1]          # close
#         radar_data[stage] = v

#     # ── FIX: explicit ylim with 20 % headroom ─────────────────
#     all_radar_vals = [v for vlist in radar_data.values() for v in vlist]
#     radar_max = max(all_radar_vals) if all_radar_vals else 1.0
#     radar_max = max(radar_max * 1.20, 0.10)   # at least 0.10 so grid shows

#     ax3.set_ylim(0, radar_max)

#     # Grid rings at 25 / 50 / 75 / 100 % of the range
#     ring_vals = [radar_max * f for f in [0.25, 0.50, 0.75, 1.00]]
#     ax3.set_yticks(ring_vals)
#     ax3.set_yticklabels([f"{v:.2f}" for v in ring_vals], fontsize=6,
#                         color="#888888")
#     ax3.yaxis.set_tick_params(labelsize=6)
#     ax3.set_rlabel_position(15)   # move radial labels away from 0° axis

#     # Draw each stage
#     for stage, c in zip(stages, colors):
#         v = radar_data[stage]
#         ax3.plot(angles, v, "o-", linewidth=2.0,
#                  label=stage, color=c, markersize=5,
#                  zorder=3)
#         ax3.fill(angles, v, alpha=0.08, color=c, zorder=2)

#     ax3.set_xticks(angles[:-1])
#     ax3.set_xticklabels(radar_labels, fontsize=9, fontweight="500")
#     ax3.set_title("Bias + performance radar\n(closer to centre = better)",
#                   fontsize=11, fontweight="bold", y=1.14)
#     ax3.legend(loc="upper right",
#                bbox_to_anchor=(1.45, 1.20),
#                fontsize=8,
#                framealpha=0.9)

#     # ── Panel 4: DI-weighted composite score ──────────────────
#     ax4 = fig.add_subplot(gs[1, 1])
#     ax4.set_facecolor("#FFFFFF")
#     threshold_line = BIAS_THRESHOLD * (2 + DI_WEIGHT + 1)
#     bar_colors = ["#55A868" if c <= threshold_line else "#C44E52"
#                   for c in comp]
#     bars = ax4.bar(range(n), comp, color=bar_colors, alpha=0.85,
#                    edgecolor="white", linewidth=0.6, width=0.50)
#     for bar, v in zip(bars, comp):
#         ax4.text(bar.get_x() + bar.get_width() / 2,
#                  bar.get_height() + 0.005,
#                  f"{v:.4f}", ha="center", va="bottom",
#                  fontsize=9, fontweight="bold", color="#222222")
#     ax4.axhline(threshold_line, color="red", linestyle="--", linewidth=1.4,
#                 label=f"Threshold ({threshold_line:.2f})")
#     best_idx = int(np.argmin(comp))
#     y_ann = comp[best_idx] + max(comp) * 0.12
#     ax4.annotate("Best", xy=(best_idx, comp[best_idx]),
#                  xytext=(best_idx, y_ann),
#                  ha="center", fontsize=10, color="#27AE60",
#                  arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
#     ax4.set_title(f"DI-weighted composite score  (DI weight = {DI_WEIGHT}×)",
#                   fontsize=11, fontweight="bold", pad=8)
#     ax4.set_ylabel("Score (lower = fairer)")
#     ax4.set_xticks(range(n))
#     ax4.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
#     ax4.legend(handles=[
#         mpatches.Patch(color="#55A868", label="Pass (fair)"),
#         mpatches.Patch(color="#C44E52", label="Fail (biased)"),
#         mpatches.Patch(color="red",     label="Threshold line"),
#     ], fontsize=8)
#     ax4.spines["top"].set_visible(False)
#     ax4.spines["right"].set_visible(False)
#     ax4.set_ylim(0, max(comp) * 1.30 + 0.05)

#     # ── Panel 5: DI raw ratio line chart ──────────────────────
#     ax5 = fig.add_subplot(gs[2, 0])
#     ax5.set_facecolor("#FFFFFF")
#     ax5.plot(range(n), di_r, "o-", color="#E24B4A", linewidth=2.2,
#              markersize=9, markerfacecolor="white", markeredgewidth=2.2,
#              label="DI (raw ratio)", zorder=4)
#     for i, v in enumerate(di_r):
#         ax5.annotate(f"{v:.3f}", (i, v),
#                      textcoords="offset points", xytext=(0, 11),
#                      ha="center", fontsize=9, color="#E24B4A")
#     ax5.axhline(1.0, color="#27AE60", linestyle="--",
#                 linewidth=1.5, label="Ideal  DI = 1.0", zorder=3)
#     ax5.axhline(0.8, color="#FF9800", linestyle=":",
#                 linewidth=1.2, label="80% rule bounds", zorder=3)
#     ax5.axhline(1.2, color="#FF9800", linestyle=":",
#                 linewidth=1.2, zorder=3)
#     ax5.fill_between(range(n), 0.8, 1.2,
#                      alpha=0.08, color="#27AE60", label="Fair zone (0.8–1.2)")
#     ax5.set_title("Disparate Impact raw ratio across stages\n"
#                   "(ideal = 1.0,  fair zone shaded green)",
#                   fontsize=11, fontweight="bold", pad=8)
#     ax5.set_xticks(range(n))
#     ax5.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
#     ax5.set_ylabel("DI ratio")
#     di_min = min(min(di_r) - 0.15, 0.70)
#     di_max = max(max(di_r) + 0.25, 1.35)
#     ax5.set_ylim(di_min, di_max)
#     ax5.legend(fontsize=8)
#     ax5.spines["top"].set_visible(False)
#     ax5.spines["right"].set_visible(False)

#     # ── Panel 6: SPD vs |1-DI| trade-off scatter ──────────────
#     ax6 = fig.add_subplot(gs[2, 1])
#     ax6.set_facecolor("#FFFFFF")
#     for i, (s, c) in enumerate(zip(stages, colors)):
#         ax6.scatter(spd[i], di_d[i], s=160, color=c, zorder=5,
#                     edgecolors="white", linewidths=1.5)
#         ax6.annotate(s, (spd[i], di_d[i]),
#                      textcoords="offset points", xytext=(9, 4),
#                      fontsize=8, color=c)
#     # arrows from baseline to each mitigation stage
#     if n > 1:
#         for i in range(1, n):
#             ax6.annotate("",
#                 xy=(spd[i], di_d[i]),
#                 xytext=(spd[0], di_d[0]),
#                 arrowprops=dict(arrowstyle="->", color=colors[i],
#                                 lw=1.1, linestyle="dashed"))
#     ax6.axhline(DI_THRESHOLD, color="#E24B4A", linestyle="--",
#                 linewidth=1.0, alpha=0.7,
#                 label=f"|1−DI| threshold ({DI_THRESHOLD})")
#     ax6.axvline(BIAS_THRESHOLD, color="#4C72B0", linestyle="--",
#                 linewidth=1.0, alpha=0.7,
#                 label=f"SPD threshold ({BIAS_THRESHOLD})")
#     ax6.fill_between([0, BIAS_THRESHOLD], 0, DI_THRESHOLD,
#                      alpha=0.09, color="#27AE60")
#     ax6.text(BIAS_THRESHOLD * 0.45, DI_THRESHOLD * 0.38,
#              "Both\nfair", ha="center", fontsize=8, color="#27AE60",
#              fontweight="bold")
#     ax6.set_title("SPD vs |1−DI| trade-off\n"
#                   "(bottom-left corner = best, green = both pass)",
#                   fontsize=11, fontweight="bold", pad=8)
#     ax6.set_xlabel("|SPD|  (lower = better)")
#     ax6.set_ylabel("|1 − DI|  (lower = better)")
#     ax6.legend(fontsize=8)
#     ax6.spines["top"].set_visible(False)
#     ax6.spines["right"].set_visible(False)

#     fig.suptitle("AIF360 Bias Detection & Mitigation — Full Report  (v4)",
#                  fontsize=15, fontweight="bold", y=1.005)

#     # ── Inline display (no separate file) ─────────────────────
#     # _show_inline(fig) //TE commented prev code to replace with following line
#     return fig #TE


# # =============================================================
# # RANKING + JUSTIFICATION
# # =============================================================
# def rank_and_justify(results):
#     print("\n" + "="*55)
#     print("  FINAL RANKING & JUSTIFICATION")
#     print("="*55)
#     scores = {s: fairness_score(m) for s, m in results.items()}
#     ranked = sorted(scores.items(), key=lambda x: x[1])

#     for i, (stage, score) in enumerate(ranked, 1):
#         flag = "🏆" if i == 1 else f"  {i}."
#         print(f"  {flag}  {stage:32s}  composite = {score:.4f}")

#     best = ranked[0][0]
#     m    = results[best]

#     print(f"\n  WHY '{best}' IS RECOMMENDED:")
#     checks = [
#         ("SPD",    abs(m["SPD"]),       "Statistical parity",  BIAS_THRESHOLD),
#         ("|1−DI|", abs(1-m["DI"]),      "Disparate impact",    DI_THRESHOLD),
#         ("EOD",    abs(m["EOD"]),        "Equal opportunity",   BIAS_THRESHOLD),
#         ("AOD",    abs(m["AOD"]),        "Average odds",        BIAS_THRESHOLD),
#     ]
#     for name, val, label, thresh in checks:
#         if val < 0.05:
#             status = "✅  very close to ideal"
#         elif val < thresh:
#             status = "✅  within threshold"
#         elif val < thresh * 2:
#             status = "⚠️  moderate bias"
#         else:
#             status = "❌  significant bias"
#         print(f"    {label:28s}  {name:8s} = {val:.4f}  {status}")

#     print(f"\n  Performance of '{best}':")
#     print(f"    Accuracy={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
#           f"Precision={m['Precision']:.4f}  Recall={m['Recall']:.4f}")

#     di_dev = abs(1 - m["DI"])
#     if di_dev > DI_THRESHOLD:
#         print(f"\n  ⚠️  DI TRADE-OFF NOTE:")
#         print(f"    |1−DI| = {di_dev:.4f} exceeds threshold.")
#         print(f"    This is expected when base rates differ across groups")
#         print(f"    (Chouldechova impossibility theorem).")
#         print(f"    Next steps:")
#         print(f"      1. Increase repair_level toward 1.0 in DIR stage.")
#         print(f"      2. Try fairlearn ExponentiatedGradient with")
#         print(f"         DemographicParity for a direct DI constraint.")
#     return best


# # =============================================================
# # MAIN PIPELINE
# # =============================================================
# # def run_bias_pipeline(): //TE commented prev line to replace with following line of code
# def run_bias_pipeline(df, target, protected, fav, priv): 

#     # 1. Load ─────────────────────────────────────────────────
#     # df = pd.read_csv(input("CSV path: ")) //TE commented prev code
#     show_columns(df)
#     if len(df) > 15_000:
#         df = df.sample(15_000, random_state=42)

#     # target, protected = get_user_columns(df) //TE commented prev code
#     df = feature_engineering(df)
#     df = preprocess_labels(df, target, protected, fav, priv) #TE added , fav, priv
#     df = clean_data(df, target, protected)

#     # 2. Diagnosis ────────────────────────────────────────────
#     diagnose_group_rates(df, target, protected)

#     X = df.drop(columns=[target])
#     y = df[target]
#     p = df[protected]

#     X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
#         X, y, p, test_size=0.3, stratify=y, random_state=42
#     )

#     scaler    = MinMaxScaler()
#     X_train_sc = scaler.fit_transform(X_train)
#     X_test_sc  = scaler.transform(X_test)

#     priv_groups   = [{protected: 1}]
#     unpriv_groups = [{protected: 0}]
#     results       = {}

#     # 3. BASELINE ─────────────────────────────────────────────
#     print("\n" + "─"*55)
#     print("  STEP 1: BASELINE MODEL")
#     print("─"*55)

#     lr = LogisticRegression(max_iter=5000)
#     lr.fit(X_train_sc, y_train)
#     y_pred_base = lr.predict(X_test_sc)

#     ds_test_true = make_ds(X_test_sc, y_test.values, p_test.values,
#                            target, protected)
#     ds_base_pred = make_ds(X_test_sc, y_pred_base,   p_test.values,
#                            target, protected)
#     cm_base      = ClassificationMetric(ds_test_true, ds_base_pred,
#                                          privileged_groups=priv_groups,
#                                          unprivileged_groups=unpriv_groups)
#     results["Baseline"] = get_metric_dict(cm_base, y_test, y_pred_base)
#     print_report(results["Baseline"], "BASELINE")

#     if not is_biased(results["Baseline"]):
#         print("\n✅  Baseline is already fair — skipping mitigations.")

#         fig = plot_results(results)
#         best = rank_and_justify(results)

#         return results, fig, best   # ✅ FIX

#     # Shared training AIF360 dataset
#     train_df_aif              = pd.DataFrame(X_train_sc)
#     train_df_aif[target]      = y_train.values
#     train_df_aif[protected]   = p_train.values
#     train_ds                  = make_dataset(train_df_aif, target, protected)

#     # 4a. SMOTE ───────────────────────────────────────────────
#     if SMOTE_AVAILABLE:
#         print("\n" + "─"*55)
#         print("  STEP 2a: PRE-PROCESSING → SMOTE")
#         print("─"*55)
#         try:
#             sm           = SMOTE(random_state=42)
#             X_sm, y_sm   = sm.fit_resample(X_train_sc, y_train)
#             p_sm_vals    = []
#             for lbl in y_sm:
#                 matching = p_train[y_train == lbl]
#                 p_sm_vals.append(
#                     np.random.choice(matching) if len(matching) > 0 else 0
#                 )
#             p_sm    = np.array(p_sm_vals)
#             lr_sm   = LogisticRegression(max_iter=5000)
#             lr_sm.fit(X_sm, y_sm)
#             y_pred_sm   = lr_sm.predict(X_test_sc)
#             ds_sm_pred  = make_ds(X_test_sc, y_pred_sm, p_test.values,
#                                   target, protected)
#             cm_sm       = ClassificationMetric(ds_test_true, ds_sm_pred,
#                                                privileged_groups=priv_groups,
#                                                unprivileged_groups=unpriv_groups)
#             results["SMOTE"] = get_metric_dict(cm_sm, y_test, y_pred_sm)
#             print_report(results["SMOTE"], "AFTER SMOTE")
#         except Exception as e:
#             print(f"  ⚠️  SMOTE failed: {e}")

#     # 4b. DisparateImpactRemover ──────────────────────────────
#     print("\n" + "─"*55)
#     print("  STEP 2b: PRE-PROCESSING → DisparateImpactRemover (repair=0.8)")
#     print("─"*55)
#     try:
#         dir_model         = DisparateImpactRemover(repair_level=0.8)
#         train_repaired    = dir_model.fit_transform(train_ds)
#         X_dir             = train_repaired.features
#         y_dir             = train_repaired.labels.ravel()
#         lr_dir            = LogisticRegression(max_iter=5000)
#         lr_dir.fit(X_dir, y_dir)

#         test_df_aif           = pd.DataFrame(X_test_sc)
#         test_df_aif[target]   = y_test.values
#         test_df_aif[protected]= p_test.values
#         test_ds_raw           = make_dataset(test_df_aif, target, protected)
#         test_repaired         = dir_model.transform(test_ds_raw)
#         y_pred_dir            = lr_dir.predict(test_repaired.features)

#         ds_dir_pred = make_ds(X_test_sc, y_pred_dir, p_test.values,
#                               target, protected)
#         cm_dir      = ClassificationMetric(ds_test_true, ds_dir_pred,
#                                            privileged_groups=priv_groups,
#                                            unprivileged_groups=unpriv_groups)
#         results["DIR (0.8)"] = get_metric_dict(cm_dir, y_test, y_pred_dir)
#         print_report(results["DIR (0.8)"], "AFTER DISPARATE IMPACT REMOVER")
#     except Exception as e:
#         print(f"  ⚠️  DisparateImpactRemover failed: {e}")

#     # 5. Reweighing ───────────────────────────────────────────
#     print("\n" + "─"*55)
#     print("  STEP 3: PRE-PROCESSING → Reweighing")
#     print("─"*55)
#     rw       = Reweighing(privileged_groups=priv_groups,
#                           unprivileged_groups=unpriv_groups)
#     train_rw = rw.fit_transform(train_ds)
#     lr_rw    = LogisticRegression(max_iter=5000)
#     lr_rw.fit(X_train_sc, y_train,
#               sample_weight=train_rw.instance_weights)
#     y_pred_rw  = lr_rw.predict(X_test_sc)
#     ds_rw_pred = make_ds(X_test_sc, y_pred_rw, p_test.values,
#                          target, protected)
#     cm_rw      = ClassificationMetric(ds_test_true, ds_rw_pred,
#                                       privileged_groups=priv_groups,
#                                       unprivileged_groups=unpriv_groups)
#     results["Reweighing"] = get_metric_dict(cm_rw, y_test, y_pred_rw)
#     print_report(results["Reweighing"], "AFTER REWEIGHING")

#     # 6. CalibratedEqOdds ─────────────────────────────────────
#     print("\n" + "─"*55)
#     print("  STEP 4: POST-PROCESSING → CalibratedEqOddsPostprocessing")
#     print("─"*55)
#     try:
#         cpp = CalibratedEqOddsPostprocessing(
#             privileged_groups=priv_groups,
#             unprivileged_groups=unpriv_groups,
#             cost_constraint="fpr"
#         )
#         cpp.fit(ds_test_true, ds_rw_pred)
#         ds_cpp      = cpp.predict(ds_rw_pred)
#         y_pred_cpp  = ds_cpp.labels.flatten().astype(int)
#         cm_cpp      = ClassificationMetric(ds_test_true, ds_cpp,
#                                            privileged_groups=priv_groups,
#                                            unprivileged_groups=unpriv_groups)
#         results["CalibratedEqOdds"] = get_metric_dict(cm_cpp, y_test, y_pred_cpp)
#         print_report(results["CalibratedEqOdds"], "AFTER CALIBRATED EQ ODDS")
#     except Exception as e:
#         print(f"  ⚠️  CalibratedEqOdds failed: {e}")

#     # # 7. Adversarial Debiasing (escalation only) ──────────────  //TE for tensor error commented prev code start
#     # best_so_far = min(results, key=lambda s: fairness_score(results[s]))
#     # if not is_biased(results[best_so_far]):
#     #     print(f"\n✅  '{best_so_far}' within threshold — skipping Adversarial.")
#     # else:
#     #     print("\n" + "─"*55)
#     #     print("  STEP 5 (ESCALATION): IN-PROCESSING → Adversarial Debiasing")
#     #     print("─"*55)
#     #     try:
#     #         tf.compat.v1.disable_eager_execution()
#     #         sess = tf.compat.v1.Session()
#     #         adv  = AdversarialDebiasing(
#     #             privileged_groups=priv_groups,
#     #             unprivileged_groups=unpriv_groups,
#     #             scope_name="adv_debiasing",
#     #             sess=sess
#     #         )
#     #         adv.fit(train_ds)
#     #         ds_test_adv = make_ds(X_test_sc, y_test.values, p_test.values,
#     #                               target, protected)
#     #         adv_pred    = adv.predict(ds_test_adv)
#     #         y_pred_adv  = adv_pred.labels.flatten().astype(int)
#     #         cm_adv      = ClassificationMetric(ds_test_true, adv_pred,
#     #                                            privileged_groups=priv_groups,
#     #                                            unprivileged_groups=unpriv_groups)
#     #         results["Adversarial"] = get_metric_dict(cm_adv, y_test, y_pred_adv)
#     #         print_report(results["Adversarial"], "AFTER ADVERSARIAL DEBIASING")
#     #         sess.close()
#     #     except Exception as e:
#     #         print(f"  ⚠️  Adversarial Debiasing failed: {e}")   //TE for tensor error commented prev code end

#     # 8. Final output ─────────────────────────────────────────
#     # plot_results(results) //TE commented prev code to replace with following one
#     # rank_and_justify(results) //TE commented prev code to replace with following one
#     fig = plot_results(results) #   TE start
#     best = rank_and_justify(results)
#     print("\n✅  PIPELINE COMPLETED")
#     return results, fig, best  #TE end
    


# # if __name__ == "__main__": //TE commented prev code
# #     run_bias_pipeline()   //TE commented prev code


# =============================================================
#  AIF360 Bias Detection & Mitigation Pipeline  v5
#  Changes from v4:
#    - Added support for custom thresholds from UI
#    - Thresholds can be passed as parameters
#    - Dynamic threshold handling throughout the pipeline
# =============================================================
# =============================================================
#  AIF360 Bias Detection & Mitigation Pipeline  v5
#  Changes from v4:
#    - Updated to use DI directly (range 0.8-1.2) instead of |1-DI|
#    - Added support for custom thresholds from UI
# =============================================================
# =============================================================
#  AIF360 Bias Detection & Mitigation Pipeline  v5
#  Changes from v4:
#    - Updated to use DI directly (range 0.8-1.2) instead of |1-DI|
#    - Added support for custom thresholds from UI
# =============================================================

import io
import pandas as pd
import numpy as np
import hashlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.inprocessing import AdversarialDebiasing

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not found — SMOTE step will be skipped.")
    print("   Install with:  pip install imbalanced-learn\n")

# Global variables for thresholds (will be updated from parameters)
BIAS_THRESHOLD = 0.10   # SPD / EOD / AOD
DI_LOWER = 0.80         # DI should be >= 0.8
DI_UPPER = 1.00         # DI should be <= 1.0
DI_WEIGHT = 1.5         # extra penalty for DI in composite score

# =============================================================
# HELPERS
# =============================================================
def safe(v):
    if v is None:
        return 0.0
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def get_metric_dict(cm, y_true, y_pred):
    return {
        "SPD":       safe(cm.statistical_parity_difference()),
        "DI":        safe(cm.disparate_impact()),
        "EOD":       safe(cm.equal_opportunity_difference()),
        "AOD":       safe(cm.average_odds_difference()),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def fairness_score(metrics):
    """DI-weighted composite — lower = fairer."""
    # Calculate DI deviation from ideal range [DI_LOWER, DI_UPPER]
    di = metrics["DI"]
    if di < DI_LOWER:
        di_deviation = DI_LOWER - di
    elif di > DI_UPPER:
        di_deviation = di - DI_UPPER
    else:
        di_deviation = 0
    
    return (
        abs(metrics["SPD"]) +
        DI_WEIGHT * di_deviation +
        abs(metrics["EOD"]) +
        abs(metrics["AOD"])
    )


def is_biased(metrics):
    di = metrics["DI"]
    di_violation = (di < DI_LOWER) or (di > DI_UPPER)
    
    return (
        abs(metrics["SPD"]) > BIAS_THRESHOLD or
        abs(metrics["EOD"]) > BIAS_THRESHOLD or
        abs(metrics["AOD"]) > BIAS_THRESHOLD or
        di_violation
    )


def print_report(metrics, stage):
    di = metrics["DI"]
    di_status = "✅" if (DI_LOWER <= di <= DI_UPPER) else "⚠️"
    
    print(f"\n{'='*55}")
    print(f"  {stage}")
    print(f"{'='*55}")
    print(f"  FAIRNESS METRICS")
    print(f"    SPD  : {metrics['SPD']:+.4f}   (ideal 0, threshold: ±{BIAS_THRESHOLD})")
    print(f"    DI   :  {metrics['DI']:.4f}   (ideal {DI_LOWER}-{DI_UPPER}) {di_status}")
    print(f"    EOD  : {metrics['EOD']:+.4f}   (ideal 0, threshold: ±{BIAS_THRESHOLD})")
    print(f"    AOD  : {metrics['AOD']:+.4f}   (ideal 0, threshold: ±{BIAS_THRESHOLD})")
    print(f"  COMPOSITE (DI-weighted) : {fairness_score(metrics):.4f}")
    print(f"  PERFORMANCE METRICS")
    print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"    F1        : {metrics['F1']:.4f}")
    print(f"    Precision : {metrics['Precision']:.4f}")
    print(f"    Recall    : {metrics['Recall']:.4f}")
    flag = "⚠️  BIAS DETECTED" if is_biased(metrics) else "✅  WITHIN THRESHOLD"
    print(f"\n  {flag}")


# =============================================================
# DATA HELPERS
# =============================================================
def show_columns(df):
    print(f"\n📊 DATASET SHAPE: {df.shape}")
    print("\n📌 COLUMNS:")
    for col in df.columns:
        print(f"   - {col}")
    print("\n📌 SAMPLE:")
    print(df.head())


def diagnose_group_rates(df, target, protected):
    print("\n" + "─"*55)
    print("  GROUP POSITIVE-RATE DIAGNOSIS")
    print("─"*55)
    rates = {}
    for g in sorted(df[protected].unique()):
        mask  = df[protected] == g
        rate  = df.loc[mask, target].mean()
        label = "PRIVILEGED" if g == 1 else "UNPRIVILEGED"
        print(f"  Group {g} ({label:12s}):  P(y=1) = {rate:.4f}  (n={mask.sum():,})")
        rates[g] = rate
    if 0 in rates and 1 in rates and rates[1] > 0:
        ratio = rates[0] / rates[1]
        print(f"\n  Base-rate ratio (unpriv ÷ priv) = {ratio:.4f}")
        if ratio < 0.1:
            print("  ⚠️  SEVERE imbalance — DI needs dataset-level fix.")
        elif ratio < 0.5:
            print("  ⚠️  Moderate imbalance — SMOTE / DIR strongly recommended.")
        else:
            print("  ✅  Reasonable balance — standard mitigation should work.")
    return rates


def feature_engineering(df):
    df = df.copy()
    for col in list(df.columns):
        if "time" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col + "_hour"] = df[col].dt.hour
            df.drop(columns=[col], inplace=True)
        elif "ip" in col.lower():
            df[col] = df[col].astype(str).apply(
                lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 1000
            )
    return df


def clean_data(df, target, protected):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    cat_cols = df.select_dtypes(include=["object"]).columns
    safe_cols = []
    high_card = []

    for col in cat_cols:
        if col == target:
            continue
        elif col == protected:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        elif df[col].nunique() <= 20:
            safe_cols.append(col)
        else:
            high_card.append(col)

    # Drop high-cardinality columns
    df = df.drop(columns=high_card)

    # Label encode safe categorical columns (instead of one-hot)
    for col in safe_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Reduce memory usage
    df = df.astype("float32")
    return df


def preprocess_labels(df, target, protected, fav, priv):
    df = df.copy()
    df[target]    = df[target].astype(str).str.lower()
    df[protected] = df[protected].astype(str).str.lower()

    print("\nTarget values:", df[target].unique())
    df[target] = df[target].apply(
        lambda x: 1 if str(x).lower() == str(fav).lower() else 0
    )

    print("\nProtected values:", df[protected].unique())
    df[protected] = df[protected].apply(
        lambda x: 1 if str(x).lower() == str(priv).lower() else 0
    )
    return df


def make_dataset(df, target, protected):
    return BinaryLabelDataset(
        df=df,
        label_names=[target],
        protected_attribute_names=[protected]
    )


def make_ds(X_arr, y_arr, p_arr, target, protected):
    df_tmp = pd.DataFrame(X_arr)
    df_tmp[target]    = np.array(y_arr)
    df_tmp[protected] = np.array(p_arr)
    return make_dataset(df_tmp, target, protected)


# =============================================================
# 6-PANEL PLOT (Updated with DI range)
# =============================================================
def plot_results(results, bias_threshold=0.10, di_lower=0.80, di_upper=1.20):
    """
    Panel 1 — Fairness deviation bars
    Panel 2 — Performance bars
    Panel 3 — Radar chart
    Panel 4 — DI-weighted composite score
    Panel 5 — DI raw ratio line chart
    Panel 6 — SPD vs DI deviation trade-off scatter
    """
    stages = list(results.keys())
    n      = len(stages)

    # colour palette — one distinct colour per stage
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2",
               "#937860", "#DA8BC3", "#64B5CD", "#CCB974"]
    colors  = palette[:n]

    # ── collect values ────────────────────────────────────────
    spd  = [abs(results[s]["SPD"])     for s in stages]
    # Calculate DI deviation from ideal range
    di_dev = []
    for s in stages:
        di = results[s]["DI"]
        if di < di_lower:
            di_dev.append(di_lower - di)
        elif di > di_upper:
            di_dev.append(di - di_upper)
        else:
            di_dev.append(0)
    
    di_r = [results[s]["DI"]           for s in stages]
    eod  = [abs(results[s]["EOD"])     for s in stages]
    aod  = [abs(results[s]["AOD"])     for s in stages]
    acc  = [results[s]["Accuracy"]     for s in stages]
    f1   = [results[s]["F1"]           for s in stages]
    prec = [results[s]["Precision"]    for s in stages]
    rec  = [results[s]["Recall"]       for s in stages]
    comp = [fairness_score(results[s]) for s in stages]

    x     = np.arange(n)
    bar_w = 0.18

    fig = plt.figure(figsize=(20, 19))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

    # ── Panel 1: Fairness deviation bars ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    m_labels = ["SPD", "DI Deviation", "EOD", "AOD"]
    m_vals   = [spd, di_dev, eod, aod]
    m_colors = ["#4C72B0", "#E24B4A", "#55A868", "#C44E52"]
    
    # Highlight threshold violations with red borders
    for i, (vals, mc, ml) in enumerate(zip(m_vals, m_colors, m_labels)):
        bars = ax1.bar(x + i * bar_w, vals, bar_w, label=ml,
                       color=mc, alpha=0.85, edgecolor="white", linewidth=0.5)
        
        # Add red border to bars that exceed threshold
        for bar_idx, (bar, v) in enumerate(zip(bars, vals)):
            if ml == "SPD" and v > bias_threshold:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
            elif ml == "DI Deviation" and v > 0:  # Any deviation is bad for DI
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
            elif ml in ["EOD", "AOD"] and v > bias_threshold:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
            
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.004,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=7, color="#222222")
    
    ax1.axhline(bias_threshold, color="red", linestyle="--",
                linewidth=1.2, label=f"SPD/EOD/AOD Threshold ({bias_threshold})")
    ax1.set_title("Fairness metrics — deviation from ideal",
                  fontsize=11, fontweight="bold", pad=8)
    ax1.set_xticks(x + bar_w * 1.5)
    ax1.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
    ax1.set_ylabel("Absolute deviation (lower = better)")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ymax = max(max(spd), max(di_dev), max(eod), max(aod))
    ax1.set_ylim(0, ymax * 1.35 + 0.02)

    # ── Panel 2: Performance bars ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    p_labels = ["Accuracy", "F1", "Precision", "Recall"]
    p_vals   = [acc, f1, prec, rec]
    p_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (vals, pc, pl) in enumerate(zip(p_vals, p_colors, p_labels)):
        bars = ax2.bar(x + i * bar_w, vals, bar_w, label=pl,
                       color=pc, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.004,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=7, color="#222222")
    ax2.set_title("Performance metrics (higher = better)",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.set_xticks(x + bar_w * 1.5)
    ax2.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Radar chart ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    ax3.set_facecolor("#FFFFFF")

    radar_labels = ["SPD", "DI Dev", "EOD", "AOD", "1−F1"]
    num_v  = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_v, endpoint=False).tolist()
    angles += angles[:1]

    # Compute all radar values
    radar_data = {}
    for stage in stages:
        r = results[stage]
        di = r["DI"]
        if di < di_lower:
            di_d = di_lower - di
        elif di > di_upper:
            di_d = di - di_upper
        else:
            di_d = 0
        v = [abs(r["SPD"]), di_d, abs(r["EOD"]), abs(r["AOD"]), 1 - r["F1"]]
        v += v[:1]
        radar_data[stage] = v

    all_radar_vals = [v for vlist in radar_data.values() for v in vlist]
    radar_max = max(all_radar_vals) if all_radar_vals else 1.0
    radar_max = max(radar_max * 1.20, 0.10)

    ax3.set_ylim(0, radar_max)
    ring_vals = [radar_max * f for f in [0.25, 0.50, 0.75, 1.00]]
    ax3.set_yticks(ring_vals)
    ax3.set_yticklabels([f"{v:.2f}" for v in ring_vals], fontsize=6, color="#888888")
    ax3.yaxis.set_tick_params(labelsize=6)
    ax3.set_rlabel_position(15)

    for stage, c in zip(stages, colors):
        v = radar_data[stage]
        ax3.plot(angles, v, "o-", linewidth=2.0,
                 label=stage, color=c, markersize=5, zorder=3)
        ax3.fill(angles, v, alpha=0.08, color=c, zorder=2)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_labels, fontsize=9, fontweight="500")
    ax3.set_title("Bias + performance radar\n(closer to centre = better)",
                  fontsize=11, fontweight="bold", y=1.14)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.45, 1.20),
               fontsize=8, framealpha=0.9)

    # ── Panel 4: DI-weighted composite score ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    threshold_line = bias_threshold * (2 + DI_WEIGHT + 1)
    bar_colors = ["#55A868" if c <= threshold_line else "#C44E52" for c in comp]
    bars = ax4.bar(range(n), comp, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.6, width=0.50)
    for bar, v in zip(bars, comp):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#222222")
    ax4.axhline(threshold_line, color="red", linestyle="--", linewidth=1.4,
                label=f"Threshold ({threshold_line:.2f})")
    best_idx = int(np.argmin(comp))
    y_ann = comp[best_idx] + max(comp) * 0.12
    ax4.annotate("Best", xy=(best_idx, comp[best_idx]),
                 xytext=(best_idx, y_ann),
                 ha="center", fontsize=10, color="#27AE60",
                 arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
    ax4.set_title(f"DI-weighted composite score  (DI weight = {DI_WEIGHT}×)",
                  fontsize=11, fontweight="bold", pad=8)
    ax4.set_ylabel("Score (lower = fairer)")
    ax4.set_xticks(range(n))
    ax4.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
    ax4.legend(handles=[
        mpatches.Patch(color="#55A868", label="Pass (fair)"),
        mpatches.Patch(color="#C44E52", label="Fail (biased)"),
        mpatches.Patch(color="red", label="Threshold line"),
    ], fontsize=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylim(0, max(comp) * 1.30 + 0.05)

    # ── Panel 5: DI raw ratio line chart ──────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("#FFFFFF")
    ax5.plot(range(n), di_r, "o-", color="#E24B4A", linewidth=2.2,
             markersize=9, markerfacecolor="white", markeredgewidth=2.2,
             label="DI (raw ratio)", zorder=4)
    for i, v in enumerate(di_r):
        ax5.annotate(f"{v:.3f}", (i, v),
                     textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=9, color="#E24B4A")
    ax5.axhline(1.0, color="#27AE60", linestyle="--",
                linewidth=1.5, label="Ideal DI = 1.0", zorder=3)
    ax5.axhline(di_lower, color="#FF9800", linestyle=":",
                linewidth=1.2, label=f"DI lower bound ({di_lower})", zorder=3)
    ax5.axhline(di_upper, color="#FF9800", linestyle=":",
                linewidth=1.2, label=f"DI upper bound ({di_upper})", zorder=3)
    ax5.fill_between(range(n), di_lower, di_upper,
                     alpha=0.08, color="#27AE60", label=f"Fair zone ({di_lower}–{di_upper})")
    ax5.set_title("Disparate Impact raw ratio across stages\n"
                  f"(ideal = 1.0, fair zone {di_lower}–{di_upper})",
                  fontsize=11, fontweight="bold", pad=8)
    ax5.set_xticks(range(n))
    ax5.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
    ax5.set_ylabel("DI ratio")
    di_min = min(min(di_r) - 0.15, di_lower - 0.1)
    di_max = max(max(di_r) + 0.25, di_upper + 0.1)
    ax5.set_ylim(di_min, di_max)
    ax5.legend(fontsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # ── Panel 6: SPD vs DI deviation trade-off scatter ────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor("#FFFFFF")
    for i, (s, c) in enumerate(zip(stages, colors)):
        ax6.scatter(spd[i], di_dev[i], s=160, color=c, zorder=5,
                    edgecolors="white", linewidths=1.5)
        ax6.annotate(s, (spd[i], di_dev[i]),
                     textcoords="offset points", xytext=(9, 4),
                     fontsize=8, color=c)
    
    if n > 1:
        for i in range(1, n):
            ax6.annotate("",
                xy=(spd[i], di_dev[i]),
                xytext=(spd[0], di_dev[0]),
                arrowprops=dict(arrowstyle="->", color=colors[i],
                                lw=1.1, linestyle="dashed"))
    
    ax6.axhline(0, color="#E24B4A", linestyle="--",
                linewidth=1.0, alpha=0.7,
                label=f"DI ideal (within {di_lower}–{di_upper})")
    ax6.axvline(bias_threshold, color="#4C72B0", linestyle="--",
                linewidth=1.0, alpha=0.7,
                label=f"SPD threshold ({bias_threshold})")
    ax6.fill_between([0, bias_threshold], 0, 0.5,
                     alpha=0.09, color="#27AE60")
    ax6.text(bias_threshold * 0.45, 0.2,
             "Both\nfair", ha="center", fontsize=8, color="#27AE60",
             fontweight="bold")
    ax6.set_title("SPD vs DI Deviation trade-off\n"
                  "(bottom-left corner = best, green = both pass)",
                  fontsize=11, fontweight="bold", pad=8)
    ax6.set_xlabel("|SPD| (lower = better)")
    ax6.set_ylabel("DI Deviation from fair range (lower = better)")
    ax6.legend(fontsize=8)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.suptitle("AIF360 Bias Detection & Mitigation — Full Report",
                 fontsize=15, fontweight="bold", y=1.005)

    return fig


# =============================================================
# RANKING + JUSTIFICATION (Updated with DI range)
# =============================================================
def rank_and_justify(results, bias_threshold=0.10, di_lower=0.80, di_upper=1.00):
    print("\n" + "="*55)
    print("  FINAL RANKING & JUSTIFICATION")
    print("="*55)
    scores = {s: fairness_score(m) for s, m in results.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1])

    for i, (stage, score) in enumerate(ranked, 1):
        flag = "🏆" if i == 1 else f"  {i}."
        print(f"  {flag}  {stage:32s}  composite = {score:.4f}")

    best = ranked[0][0]
    m    = results[best]

    print(f"\n  WHY '{best}' IS RECOMMENDED:")
    
    # SPD check
    spd_val = abs(m["SPD"])
    spd_status = "✅ within threshold" if spd_val <= bias_threshold else "❌ exceeds threshold"
    
    # DI check
    di_val = m["DI"]
    if di_lower <= di_val <= di_upper:
        di_status = f"✅ within range ({di_lower}–{di_upper})"
    elif di_val < di_lower:
        di_status = f"⚠️ below lower bound (needs ≥ {di_lower})"
    else:
        di_status = f"⚠️ above upper bound (needs ≤ {di_upper})"
    
    # EOD check
    eod_val = abs(m["EOD"])
    eod_status = "✅ within threshold" if eod_val <= bias_threshold else "❌ exceeds threshold"
    
    # AOD check
    aod_val = abs(m["AOD"])
    aod_status = "✅ within threshold" if aod_val <= bias_threshold else "❌ exceeds threshold"
    
    print(f"    SPD  = {spd_val:.4f}  {spd_status}")
    print(f"    DI   = {di_val:.4f}  {di_status}")
    print(f"    EOD  = {eod_val:.4f}  {eod_status}")
    print(f"    AOD  = {aod_val:.4f}  {aod_status}")

    print(f"\n  Performance of '{best}':")
    print(f"    Accuracy={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"Precision={m['Precision']:.4f}  Recall={m['Recall']:.4f}")

    if not (di_lower <= di_val <= di_upper):
        print(f"\n  ⚠️  DI RANGE NOTE:")
        print(f"    DI = {di_val:.4f} is outside the fair range [{di_lower}–{di_upper}].")
        print(f"    This happens when base rates differ significantly across groups.")
        print(f"    Next steps:")
        print(f"      1. Increase repair_level toward 1.0 in DIR stage.")
        print(f"      2. Try SMOTE oversampling to balance the dataset.")
        print(f"      3. Use Reweighing to adjust sample weights.")
    return best


# =============================================================
# MAIN PIPELINE (Updated with DI range parameters)
# =============================================================
def run_bias_pipeline(df, target, protected, fav, priv, custom_thresholds=None):
    """
    Run bias detection and mitigation pipeline
    
    Parameters:
    - df: pandas DataFrame with the dataset
    - target: name of target column
    - protected: name of protected attribute column
    - fav: favorable label value
    - priv: privileged group value
    - custom_thresholds: dict with keys 'SPD', 'DI_lower', 'DI_upper', 'EOD', 'AOD'
                         If None, default thresholds are used (SPD=0.10, DI=0.8-1.2, EOD=0.10, AOD=0.10)
    """
    global BIAS_THRESHOLD, DI_LOWER, DI_UPPER
    
    # Set thresholds based on custom_thresholds parameter
    if custom_thresholds:
        BIAS_THRESHOLD = custom_thresholds.get('SPD', 0.10)
        DI_LOWER = custom_thresholds.get('DI_lower', 0.80)
        DI_UPPER = custom_thresholds.get('DI_upper', 1.00)
        print(f"\n📊 Using custom thresholds:")
        print(f"   SPD/EOD/AOD threshold: {BIAS_THRESHOLD}")
        print(f"   DI fair range: {DI_LOWER} - {DI_UPPER}")
    else:
        BIAS_THRESHOLD = 0.10
        DI_LOWER = 0.80
        DI_UPPER = 1.00
        print(f"\n📊 Using default thresholds:")
        print(f"   SPD/EOD/AOD threshold: {BIAS_THRESHOLD}")
        print(f"   DI fair range: {DI_LOWER} - {DI_UPPER}")

    # 1. Load and preprocess
    show_columns(df)
    if len(df) > 15000:
        df = df.sample(15000, random_state=42)

    df = feature_engineering(df)
    df = preprocess_labels(df, target, protected, fav, priv)
    df = clean_data(df, target, protected)

    # 2. Diagnosis
    diagnose_group_rates(df, target, protected)

    X = df.drop(columns=[target])
    y = df[target]
    p = df[protected]

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, p, test_size=0.3, stratify=y, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    priv_groups = [{protected: 1}]
    unpriv_groups = [{protected: 0}]
    results = {}

    # 3. BASELINE
    print("\n" + "─"*55)
    print("  STEP 1: BASELINE MODEL")
    print("─"*55)

    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_sc, y_train)
    y_pred_base = lr.predict(X_test_sc)

    ds_test_true = make_ds(X_test_sc, y_test.values, p_test.values,
                           target, protected)
    ds_base_pred = make_ds(X_test_sc, y_pred_base, p_test.values,
                           target, protected)
    cm_base = ClassificationMetric(ds_test_true, ds_base_pred,
                                   privileged_groups=priv_groups,
                                   unprivileged_groups=unpriv_groups)
    results["Baseline"] = get_metric_dict(cm_base, y_test, y_pred_base)
    print_report(results["Baseline"], "BASELINE")

    if not is_biased(results["Baseline"]):
        print("\n✅ Baseline is already fair — skipping mitigations.")
        fig = plot_results(results, BIAS_THRESHOLD, DI_LOWER, DI_UPPER)
        best = rank_and_justify(results, BIAS_THRESHOLD, DI_LOWER, DI_UPPER)
        return results, fig, best

    # Shared training AIF360 dataset
    train_df_aif = pd.DataFrame(X_train_sc)
    train_df_aif[target] = y_train.values
    train_df_aif[protected] = p_train.values
    train_ds = make_dataset(train_df_aif, target, protected)

    # 4a. SMOTE
    if SMOTE_AVAILABLE:
        print("\n" + "─"*55)
        print("  STEP 2a: PRE-PROCESSING → SMOTE")
        print("─"*55)
        try:
            sm = SMOTE(random_state=42)
            X_sm, y_sm = sm.fit_resample(X_train_sc, y_train)
            p_sm_vals = []
            for lbl in y_sm:
                matching = p_train[y_train == lbl]
                p_sm_vals.append(
                    np.random.choice(matching) if len(matching) > 0 else 0
                )
            p_sm = np.array(p_sm_vals)
            lr_sm = LogisticRegression(max_iter=5000)
            lr_sm.fit(X_sm, y_sm)
            y_pred_sm = lr_sm.predict(X_test_sc)
            ds_sm_pred = make_ds(X_test_sc, y_pred_sm, p_test.values,
                                 target, protected)
            cm_sm = ClassificationMetric(ds_test_true, ds_sm_pred,
                                         privileged_groups=priv_groups,
                                         unprivileged_groups=unpriv_groups)
            results["SMOTE"] = get_metric_dict(cm_sm, y_test, y_pred_sm)
            print_report(results["SMOTE"], "AFTER SMOTE")
        except Exception as e:
            print(f"  ⚠️ SMOTE failed: {e}")

    # 4b. DisparateImpactRemover
    print("\n" + "─"*55)
    print("  STEP 2b: PRE-PROCESSING → DisparateImpactRemover (repair=0.8)")
    print("─"*55)
    try:
        dir_model = DisparateImpactRemover(repair_level=0.8)
        train_repaired = dir_model.fit_transform(train_ds)
        X_dir = train_repaired.features
        y_dir = train_repaired.labels.ravel()
        lr_dir = LogisticRegression(max_iter=5000)
        lr_dir.fit(X_dir, y_dir)

        test_df_aif = pd.DataFrame(X_test_sc)
        test_df_aif[target] = y_test.values
        test_df_aif[protected] = p_test.values
        test_ds_raw = make_dataset(test_df_aif, target, protected)
        test_repaired = dir_model.transform(test_ds_raw)
        y_pred_dir = lr_dir.predict(test_repaired.features)

        ds_dir_pred = make_ds(X_test_sc, y_pred_dir, p_test.values,
                              target, protected)
        cm_dir = ClassificationMetric(ds_test_true, ds_dir_pred,
                                      privileged_groups=priv_groups,
                                      unprivileged_groups=unpriv_groups)
        results["DIR (0.8)"] = get_metric_dict(cm_dir, y_test, y_pred_dir)
        print_report(results["DIR (0.8)"], "AFTER DISPARATE IMPACT REMOVER")
    except Exception as e:
        print(f"  ⚠️ DisparateImpactRemover failed: {e}")

    # 5. Reweighing
    print("\n" + "─"*55)
    print("  STEP 3: PRE-PROCESSING → Reweighing")
    print("─"*55)
    rw = Reweighing(privileged_groups=priv_groups,
                    unprivileged_groups=unpriv_groups)
    train_rw = rw.fit_transform(train_ds)
    lr_rw = LogisticRegression(max_iter=5000)
    lr_rw.fit(X_train_sc, y_train,
              sample_weight=train_rw.instance_weights)
    y_pred_rw = lr_rw.predict(X_test_sc)
    ds_rw_pred = make_ds(X_test_sc, y_pred_rw, p_test.values,
                         target, protected)
    cm_rw = ClassificationMetric(ds_test_true, ds_rw_pred,
                                 privileged_groups=priv_groups,
                                 unprivileged_groups=unpriv_groups)
    results["Reweighing"] = get_metric_dict(cm_rw, y_test, y_pred_rw)
    print_report(results["Reweighing"], "AFTER REWEIGHING")

    # 6. CalibratedEqOdds
    print("\n" + "─"*55)
    print("  STEP 4: POST-PROCESSING → CalibratedEqOddsPostprocessing")
    print("─"*55)
    try:
        cpp = CalibratedEqOddsPostprocessing(
            privileged_groups=priv_groups,
            unprivileged_groups=unpriv_groups,
            cost_constraint="fpr"
        )
        cpp.fit(ds_test_true, ds_rw_pred)
        ds_cpp = cpp.predict(ds_rw_pred)
        y_pred_cpp = ds_cpp.labels.flatten().astype(int)
        cm_cpp = ClassificationMetric(ds_test_true, ds_cpp,
                                      privileged_groups=priv_groups,
                                      unprivileged_groups=unpriv_groups)
        results["CalibratedEqOdds"] = get_metric_dict(cm_cpp, y_test, y_pred_cpp)
        print_report(results["CalibratedEqOdds"], "AFTER CALIBRATED EQ ODDS")
    except Exception as e:
        print(f"  ⚠️ CalibratedEqOdds failed: {e}")

    # Final output
    fig = plot_results(results, BIAS_THRESHOLD, DI_LOWER, DI_UPPER)
    best = rank_and_justify(results, BIAS_THRESHOLD, DI_LOWER, DI_UPPER)
    print("\n✅ PIPELINE COMPLETED")
    return results, fig, best
