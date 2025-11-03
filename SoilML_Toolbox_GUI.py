#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance & Parity Plot GUI (XGBoost + SHAP)
- Tab 1: Feature importance (SHAP-NFI vs XGB Importance)
- Tab 2: Predicted vs Measured parity plot (45° reference) with metrics
- CSV loader + target selection
- Auto regression/classification (or manual override)
- Export CSV for importance; Save figures for both tabs

Requirements:
  pip install pandas numpy matplotlib xgboost shap scikit-learn
"""

import os
import threading
import traceback
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ML/Explainability
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
import shap


def is_classification(y: pd.Series, max_classes: int = 20):
    """Heuristic to decide classification vs regression."""
    if y.dtype.kind in ("i", "u", "b"):
        return y.nunique(dropna=True) <= max_classes
    if y.dtype.name == "category":
        return True
    return False


def compute_shap_nfi(model, X_df: pd.DataFrame):
    """Compute SHAP values and NFI from absolute sums (Eq. 14)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        abs_sum = None
        for sv in shap_values:
            v = np.abs(sv)
            abs_sum = v if abs_sum is None else abs_sum + v
    else:
        abs_sum = np.abs(shap_values)  # N x M

    per_feature_sum = abs_sum.sum(axis=0)  # length M
    total_sum = per_feature_sum.sum()
    if total_sum == 0:
        nfi = np.zeros_like(per_feature_sum)
    else:
        nfi = per_feature_sum / total_sum
    return pd.Series(nfi, index=X_df.columns, name="SHAP_NFI")


def get_xgb_importance(model, feature_names):
    """Return XGBoost importance aligned with encoded feature names.
    Priority: sklearn wrapper's feature_importances_ (ordered),
    Fallback: booster.get_score with f0.. mapping. Output normalized.
    """
    # 1) Try feature_importances_
    try:
        arr = model.feature_importances_
        if arr is not None and len(arr) == len(feature_names):
            imp = pd.Series(arr, index=feature_names, dtype=float)
            s = imp.sum()
            if s > 0:
                imp = imp / s
            return imp.rename("XGB_Importance")
    except Exception:
        pass

    # 2) Fallback with booster.get_score
    try:
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type="gain")
        if not score_dict:
            score_dict = booster.get_score(importance_type="weight")
        mapping = {f"f{i}": fn for i, fn in enumerate(feature_names)}
        ser = pd.Series(0.0, index=feature_names, dtype=float)
        for k, v in score_dict.items():
            fn = mapping.get(k)
            if fn is not None:
                ser[fn] = float(v)
        s = ser.sum()
        if s > 0:
            ser = ser / s
        return ser.rename("XGB_Importance")
    except Exception:
        return pd.Series(0.0, index=feature_names, name="XGB_Importance")


class NfiParityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Feature Importance & Parity Plot (XGBoost + SHAP)")
        self.geometry("1280x780")
        self.minsize(1100, 700)

        # state
        self.data = None
        self.feature_names = None
        self.results_df = None
        self.is_cls = False
        self.y_test = None
        self.y_pred = None

        # Figures
        self.fig_imp = None
        self.ax_imp = None
        self.canvas_imp = None

        self.fig_par = None
        self.ax_par = None
        self.canvas_par = None

        self._build_ui()

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="Open CSV…", command=self.on_open).pack(side=tk.LEFT)
        ttk.Label(top, text="Target:").pack(side=tk.LEFT, padx=(12, 4))
        self.cmb_target = ttk.Combobox(top, state="readonly", width=28)
        self.cmb_target.pack(side=tk.LEFT)

        ttk.Label(top, text="Test size:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_test_size = tk.StringVar(value="0.2")
        ttk.Entry(top, textvariable=self.var_test_size, width=6).pack(side=tk.LEFT)

        ttk.Label(top, text="Random seed:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_seed = tk.StringVar(value="42")
        ttk.Entry(top, textvariable=self.var_seed, width=8).pack(side=tk.LEFT)

        ttk.Label(top, text="Model type:").pack(side=tk.LEFT, padx=(12, 4))
        self.cmb_model = ttk.Combobox(top, state="readonly", width=16,
                                      values=["Auto", "Regression", "Classification"])
        self.cmb_model.current(0)
        self.cmb_model.pack(side=tk.LEFT)

        ttk.Label(top, text="Top-k to plot:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_topk = tk.StringVar(value="20")
        ttk.Entry(top, textvariable=self.var_topk, width=6).pack(side=tk.LEFT)

        ttk.Button(top, text="Run", command=self.on_run).pack(side=tk.LEFT, padx=12)

        # Tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Tab 1: Importance
        tab1 = ttk.Frame(self.nb)
        self.nb.add(tab1, text="Feature Importance")

        pan1 = ttk.Panedwindow(tab1, orient=tk.HORIZONTAL)
        pan1.pack(fill=tk.BOTH, expand=True)

        # left: table
        left1 = ttk.Frame(pan1)
        pan1.add(left1, weight=1)
        self.tree = ttk.Treeview(left1, columns=("feat", "shap", "xgb"),
                                 show="headings", selectmode="browse")
        self.tree.heading("feat", text="Feature")
        self.tree.heading("shap", text="SHAP_NFI")
        self.tree.heading("xgb", text="XGB_Importance")
        self.tree.column("feat", stretch=True, width=320)
        self.tree.column("shap", anchor=tk.E, width=120)
        self.tree.column("xgb", anchor=tk.E, width=140)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb1 = ttk.Scrollbar(left1, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb1.set)
        vsb1.pack(side=tk.RIGHT, fill=tk.Y)

        # right: importance plot + buttons
        right1 = ttk.Frame(pan1)
        pan1.add(right1, weight=1)

        self.fig_imp = plt.Figure(figsize=(6,5), dpi=100)
        self.ax_imp = self.fig_imp.add_subplot(111)
        self.ax_imp.set_title("Feature Importance (SHAP-NFI vs XGB Importance)")
        self.ax_imp.set_xlabel("Feature")
        self.ax_imp.set_ylabel("Value")

        self.canvas_imp = FigureCanvasTkAgg(self.fig_imp, master=right1)
        self.canvas_imp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btns1 = ttk.Frame(right1)
        btns1.pack(fill=tk.X, pady=(8,0))
        ttk.Button(btns1, text="Export CSV", command=self.on_export_csv).pack(side=tk.LEFT, padx=(0,12))
        ttk.Button(btns1, text="Save Figure", command=self.on_save_fig_imp).pack(side=tk.LEFT)

        # Tab 2: Parity plot
        tab2 = ttk.Frame(self.nb)
        self.nb.add(tab2, text="Prediction vs Measured")

        # plot
        self.fig_par = plt.Figure(figsize=(7,5), dpi=100)
        self.ax_par = self.fig_par.add_subplot(111)
        self.ax_par.set_title("Predicted vs Measured (Parity Plot)")
        self.ax_par.set_xlabel("Measured")
        self.ax_par.set_ylabel("Predicted")

        self.canvas_par = FigureCanvasTkAgg(self.fig_par, master=tab2)
        self.canvas_par.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btns2 = ttk.Frame(tab2)
        btns2.pack(fill=tk.X, pady=(8,0))
        ttk.Button(btns2, text="Save Parity Figure", command=self.on_save_fig_par).pack(side=tk.LEFT)

        # Status
        self.status = tk.StringVar(value="Load a CSV to begin.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0,6))

    def set_status(self, text):
        self.status.set(text)
        self.update_idletasks()

    def on_open(self):
        path = filedialog.askopenfilename(title="Open CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError("Need at least two columns (features + target).")
            self.data = df
            self.cmb_target["values"] = list(df.columns)
            self.cmb_target.set(df.columns[-1])
            self.set_status(f"Loaded: {os.path.basename(path)}  (shape: {df.shape[0]} x {df.shape[1]})")
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))
            self.set_status("Failed to load CSV.")

    def _prepare_xy(self):
        if self.data is None:
            raise RuntimeError("No data loaded.")
        tgt = self.cmb_target.get()
        if not tgt or tgt not in self.data.columns:
            raise RuntimeError("Please select a valid target column.")
        y = self.data[tgt]
        X = self.data.drop(columns=[tgt])

        # Identify categorical vs numeric
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Preprocess: impute + one-hot
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_cols),
                                                       ("cat", categorical_transformer, cat_cols)])
        return X, y, preprocessor, num_cols, cat_cols

    def on_run(self):
        def _run():
            try:
                self.set_status("Running…")
                test_size = float(self.var_test_size.get())
                seed = int(self.var_seed.get())

                X, y, preprocessor, num_cols, cat_cols = self._prepare_xy()

                # Determine model type
                mode = self.cmb_model.get()
                if mode == "Auto":
                    cls_flag = is_classification(y)
                elif mode == "Classification":
                    cls_flag = True
                else:
                    cls_flag = False
                self.is_cls = cls_flag

                Xtrain, Xtest, ytrain, ytest = train_test_split(
                    X, y, test_size=test_size, random_state=seed, stratify=y if cls_flag else None
                )

                # Fit pipeline with XGB
                if cls_flag:
                    base_model = XGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=0,
                        objective="binary:logistic" if ytrain.nunique() <= 2 else "multi:softprob"
                    )
                else:
                    base_model = XGBRegressor(
                        n_estimators=600,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=0,
                        objective="reg:squarederror"
                    )

                model = Pipeline(steps=[("prep", preprocessor), ("xgb", base_model)])
                model.fit(Xtrain, ytrain)

                # Extract feature names after preprocessing
                prep = model.named_steps["prep"]
                if len(prep.transformers_) > 0:
                    cat_transformer = prep.named_transformers_["cat"]
                    if cat_transformer is not None and len(cat_cols) > 0:
                        ohe = cat_transformer.named_steps["onehot"]
                        ohe_names = list(ohe.get_feature_names_out(cat_cols))
                    else:
                        ohe_names = []
                    feature_names = list([*num_cols, *ohe_names])
                else:
                    feature_names = list(X.columns)

                # Predictions (store for parity plot)
                Xtest_tr = prep.transform(Xtest)
                xgb = model.named_steps["xgb"]

                if cls_flag:
                    ypred = xgb.predict(Xtest_tr)
                    metric_main = accuracy_score(ytest, ypred)
                    metric_label = "Accuracy"
                else:
                    ypred = xgb.predict(Xtest_tr)
                    metric_main = r2_score(ytest, ypred)
                    metric_label = "R²"

                # Save for parity plot
                self.y_test = np.array(ytest).astype(float) if not cls_flag else np.array(ytest)
                self.y_pred = np.array(ypred).astype(float) if not cls_flag else np.array(ypred)

                # Importance on ALL data
                X_all_enc = pd.DataFrame(prep.transform(X), columns=feature_names)
                shap_nfi = compute_shap_nfi(xgb, X_all_enc)
                xgb_imp  = get_xgb_importance(xgb, feature_names)
                combined = pd.concat([shap_nfi, xgb_imp], axis=1).fillna(0.0).sort_values("SHAP_NFI", ascending=False)
                self.results_df = combined
                self.feature_names = feature_names

                # ===== Tab 1: update table + importance plot =====
                for row in self.tree.get_children():
                    self.tree.delete(row)
                for feat, row in combined.iterrows():
                    self.tree.insert("", tk.END, values=(feat, f"{row['SHAP_NFI']:.4f}", f"{row['XGB_Importance']:.4f}"))

                # draw importance
                self.ax_imp.clear()
                self.ax_imp.set_title("Feature Importance (SHAP-NFI vs XGB Importance)")
                self.ax_imp.set_xlabel("Feature")
                self.ax_imp.set_ylabel("Value")

                try:
                    topk = max(1, int(self.var_topk.get()))
                except Exception:
                    topk = 20
                sub = combined.head(topk).copy()
                feats = sub.index.to_list()
                idx = np.arange(len(feats)); width = 0.42

                vals_shap = sub["SHAP_NFI"].values
                vals_xgb  = sub["XGB_Importance"].values

                self.ax_imp.bar(idx - width/2, vals_shap, width=width, label="SHAP_NFI", align='center')
                self.ax_imp.bar(idx + width/2, vals_xgb,  width=width, label="XGB_Importance", align='center')

                try:
                    self.ax_imp.set_xticks(idx, feats)
                except TypeError:
                    self.ax_imp.set_xticks(idx); self.ax_imp.set_xticklabels(feats)
                for lab in self.ax_imp.get_xticklabels():
                    lab.set_rotation(60); lab.set_ha('right')
                self.ax_imp.set_xlim(-0.5, len(idx)-0.5)
                self.ax_imp.margins(x=0.02)
                self.ax_imp.legend()
                self.fig_imp.subplots_adjust(bottom=0.30, left=0.08, right=0.98, top=0.92)
                self.canvas_imp.draw()

                # ===== Tab 2: update parity plot =====
                self.ax_par.clear()
                self.ax_par.set_title("Predicted vs Measured (Parity Plot)")
                self.ax_par.set_xlabel("Measured")
                self.ax_par.set_ylabel("Predicted")

                x = np.array(self.y_test).astype(float) if not cls_flag else np.array(self.y_test, dtype=float)
                y = np.array(self.y_pred).astype(float) if not cls_flag else np.array(self.y_pred, dtype=float)

                self.ax_par.scatter(x, y, s=18)
                lo = np.nanmin([x.min(), y.min()])
                hi = np.nanmax([x.max(), y.max()])
                self.ax_par.plot([lo, hi], [lo, hi])

                if cls_flag:
                    txt = f"Accuracy = {metric_main:.4f}"
                else:
                    rmse = np.sqrt(mean_squared_error(x, y))
                    mae  = mean_absolute_error(x, y)
                    txt  = f"{metric_label} = {metric_main:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}"
                self.ax_par.text(0.02, 0.98, txt, transform=self.ax_par.transAxes, va='top')

                self.fig_par.tight_layout()
                self.canvas_par.draw()

                self.set_status(f"Done. {metric_label}: {metric_main:.4f} — {len(combined)} encoded features.")

            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Run failed", str(e))
                self.set_status("Run failed. See error message.")

        threading.Thread(target=_run, daemon=True).start()

    def on_export_csv(self):
        if self.results_df is None:
            messagebox.showinfo("No results", "Please run the analysis first.")
            return
        path = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv",
                                            filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            out = self.results_df.copy()
            out.index.name = "Feature"
            out.reset_index().to_csv(path, index=False)
            self.set_status(f"Saved CSV: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_save_fig_imp(self):
        if self.results_df is None:
            messagebox.showinfo("No figure", "Please run the analysis first.")
            return
        path = filedialog.asksaveasfilename(title="Save Importance Figure", defaultextension=".png",
                                            filetypes=[("PNG Image", "*.png")])
        if not path:
            return
        try:
            self.fig_imp.savefig(path, dpi=200, bbox_inches="tight")
            self.set_status(f"Saved figure: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_save_fig_par(self):
        if self.y_test is None or self.y_pred is None:
            messagebox.showinfo("No parity plot", "Please run the analysis first.")
            return
        path = filedialog.asksaveasfilename(title="Save Parity Figure", defaultextension=".png",
                                            filetypes=[("PNG Image", "*.png")])
        if not path:
            return
        try:
            self.fig_par.savefig(path, dpi=200, bbox_inches="tight")
            self.set_status(f"Saved parity figure: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


if __name__ == "__main__":
    app = NfiParityApp()
    app.mainloop()
