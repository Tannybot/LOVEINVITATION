import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class DecisionTreeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Decision Tree Analysis")
        self.root.geometry("1300x900")

        self.data = None
        self.features = []
        self.target_var = None

        # --- Decision Tree parameters ---
        self.max_depth = tk.IntVar(value=None)
        self.criterion = tk.StringVar(value='mse')
        self.model = None
        self.scaler = StandardScaler()

        # Widgets
        self._build_controls()
        self._build_preview()
        self._build_report()

        self.root.mainloop()

    def _build_controls(self):
        frm = ttk.LabelFrame(self.root, text="Controls", padding=10)
        frm.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        frm.grid_columnconfigure(0, weight=1)

        ttk.Button(frm, text="Load CSV", command=self.load_data).grid(sticky="ew", pady=5)

        # target & features
        tgt = ttk.LabelFrame(frm, text="Target Variable", padding=5)
        tgt.grid(sticky="ew", pady=5)
        self.tgt_combo = ttk.Combobox(tgt, state="readonly")
        self.tgt_combo.pack(fill="x")
        self.tgt_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_feature_list())

        feat_frame = ttk.LabelFrame(frm, text="Features", padding=5)
        feat_frame.grid(sticky="nsew", pady=5)
        canvas = tk.Canvas(feat_frame)
        scrollbar = ttk.Scrollbar(feat_frame, orient="vertical", command=canvas.yview)
        self.feat_container = ttk.Frame(canvas)
        self.feat_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.feat_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Decision Tree options
        opts = ttk.LabelFrame(frm, text="Tree Options", padding=5)
        opts.grid(sticky="ew", pady=5)
        ttk.Label(opts, text="Max depth (blank=none):").pack(anchor="w")
        ttk.Entry(opts, textvariable=self.max_depth).pack(fill="x")
        ttk.Label(opts, text="Criterion:").pack(anchor="w")
        ttk.OptionMenu(opts, self.criterion, 'mse', 'mse', 'friedman_mse', 'mae').pack(fill="x")

        ttk.Button(frm, text="Train Decision Tree", command=self.train).grid(sticky="ew", pady=5)

    def _build_preview(self):
        viz = ttk.LabelFrame(self.root, text="Data Preview", padding=10)
        viz.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        viz.grid_rowconfigure(2, weight=1)
        viz.grid_columnconfigure(0, weight=1)

        self.info_lbl = ttk.Label(viz, text="No data loaded")
        self.info_lbl.grid(sticky="ew")

        self.tree = ttk.Treeview(viz, show="headings")
        vsb = ttk.Scrollbar(viz, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(viz, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=10)

    def _build_report(self):
        rpt = ttk.LabelFrame(self.root, text="Module 5 Report", padding=10)
        rpt.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        rpt.grid_columnconfigure(0, weight=1)

        # Four sections
        for title in ["1. Project Description", "2. Code", "3. Output", 
                      "4. Visualization", "5. Insights"]:
            lbl = ttk.Label(rpt, text=title, font=("Segoe UI", 10, "bold"))
            lbl.pack(anchor="w", pady=(5,0))
            txt = tk.Text(rpt, height=3, wrap="word")
            txt.pack(fill="x", pady=(0,5))
            setattr(self, title.split()[1].lower() + "_txt", txt)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.data = pd.read_csv(path)
        # simple fillna + encode
        for c in self.data.select_dtypes(include='number'):
            self.data[c].fillna(self.data[c].mean(), inplace=True)
        for c in self.data.select_dtypes(include='object'):
            self.data[c].fillna(self.data[c].mode()[0], inplace=True)
            self.data[c] = LabelEncoder().fit_transform(self.data[c])

        # reload preview
        self._refresh_preview()
        # setup target dropdown
        cols = list(self.data.columns)
        self.tgt_combo['values'] = cols
        self.tgt_combo.set('')

    def _refresh_preview(self):
        df = self.data
        self.info_lbl['text'] = f"{df.shape[0]} rows × {df.shape[1]} cols"
        self.tree.delete(*self.tree.get_children())
        self.tree['columns'] = list(df.columns)
        for c in df.columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")
        for _, row in df.head(20).iterrows():
            self.tree.insert("", "end", values=list(row))

    def _refresh_feature_list(self):
        # clear
        for w in self.feat_container.winfo_children():
            w.destroy()
        self.features = []
        tgt = self.tgt_combo.get()
        for col in self.data.columns:
            if col == tgt: continue
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.feat_container, text=col, variable=var,
                                 command=lambda c=col, v=var: self._toggle_feature(c, v))
            cb.pack(anchor="w")
        self.target_var = tgt

    def _toggle_feature(self, col, var):
        if var.get(): self.features.append(col)
        else:          self.features.remove(col)

    def train(self):
        if not self.data or not self.features or not self.target_var:
            messagebox.showerror("Error", "Load data, select target & features first")
            return

        X = self.data[self.features]
        y = self.data[self.target_var]

        # optional normalization
        X = self.scaler.fit_transform(X)

        # build model
        md = self.max_depth.get() or None
        self.model = DecisionTreeRegressor(max_depth=md,
                                           criterion=self.criterion.get(),
                                           random_state=42)

        # timing
        t0 = time.time()
        self.model.fit(X, y)
        train_time = time.time() - t0

        # metrics
        preds = self.model.predict(X)
        r2 = r2_score(y, preds)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)

        # cross-val
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(self.model, X, y, cv=kf, scoring='r2').mean()

        # --- Populate Module 5 Report ---
        # 1. Project Description
        desc = (
            "This application allows you to load any CSV, select predictors and a target "
            "and train a Decision Tree Regressor.  "
            f"Tree parameters: max_depth={md}, criterion={self.criterion.get()}."
        )
        self.project_txt.delete("1.0","end");  self.project_txt.insert("1.0", desc)

        # 2. Code
        snippet = (
            "model = DecisionTreeRegressor(max_depth=..., criterion='...')\n"
            "model.fit(X_train, y_train)\n"
            "preds = model.predict(X_test)"
        )
        self.code_txt.delete("1.0","end");     self.code_txt.insert("1.0", snippet)

        # 3. Output
        out = (
            f"Train time: {train_time:.3f}s\n"
            f"R² (train): {r2:.4f}\n"
            f"MSE (train): {mse:.4f}\n"
            f"MAE (train): {mae:.4f}\n"
            f"5-fold CV R²: {cv_r2:.4f}"
        )
        self.output_txt.delete("1.0","end");   self.output_txt.insert("1.0", out)

        # 4. Visualization
        self.ax.clear()
        plot_tree(self.model, feature_names=self.features, filled=True, ax=self.ax)
        self.ax.set_title("Decision Tree Structure")
        self.canvas.draw()
        self.visualization_txt.delete("1.0","end")
        self.visualization_txt.insert("1.0", "Tree plotted above.")

        # 5. Insights
        # grab feature importances
        imp = sorted(zip(self.features, self.model.feature_importances_), key=lambda x:-x[1])
        insights = "Top features:\n" + "\n".join(f" - {f}: {imp:.2%}" for f,imp in imp[:5])
        self.insights_txt.delete("1.0","end"); self.insights_txt.insert("1.0", insights)

# run it
if __name__ == "__main__":
    DecisionTreeApp()
