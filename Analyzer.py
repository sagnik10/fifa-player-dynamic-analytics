import os
import time
import zipfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

START_TIME = time.time()
BASE_DIR = os.getcwd()
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"

if os.path.exists(KAGGLE_WORKING):
    BASE_DIR = KAGGLE_WORKING

OUT = os.path.join(BASE_DIR, "Output")
CHART_DIR = os.path.join(OUT, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUT, "FIFA_Dynamic_Player_Analytics_Report.pdf")
INSIGHT_CSV_PATH = os.path.join(OUT, "dynamic_insights_summary.csv")


def find_data_files():
    """Find CSV/Excel files locally first, then Kaggle input files when present."""
    search_roots = [os.getcwd()]
    if os.path.exists(KAGGLE_INPUT):
        search_roots.append(KAGGLE_INPUT)

    data_files = []
    for root in search_roots:
        for current_root, _, files in os.walk(root):
            if os.path.abspath(current_root).startswith(os.path.abspath(OUT)):
                continue

            for file_name in files:
                path = os.path.join(current_root, file_name)
                lower_name = file_name.lower()

                if lower_name.endswith((".csv", ".xlsx", ".xls")):
                    data_files.append(path)
                elif lower_name.endswith(".zip"):
                    extract_dir = os.path.join(OUT, "unzipped")
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(path, "r") as archive:
                        archive.extractall(extract_dir)
                    for zip_root, _, zip_files in os.walk(extract_dir):
                        for zip_file in zip_files:
                            if zip_file.lower().endswith((".csv", ".xlsx", ".xls")):
                                data_files.append(os.path.join(zip_root, zip_file))

    return sorted(set(data_files))


def clean_columns(frame):
    frame = frame.copy()
    frame.columns = [str(col).lower().strip().replace(" ", "_") for col in frame.columns]
    return frame


def read_dataset(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def safe_filename(value):
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in str(value))[:120]


def add_chart(charts, title, insight, path):
    if path and os.path.exists(path):
        charts.append({"title": title, "insight": insight, "path": path})


def save_fig(fig, file_name):
    path = os.path.join(CHART_DIR, safe_filename(file_name) + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def as_money(value):
    return f"EUR {value:,.2f}m"


def numeric_summary(series):
    clean = series.dropna()
    return {
        "min": clean.min(),
        "max": clean.max(),
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(),
        "skew": clean.skew(),
    }


def distribution_insight(df, col):
    stats = numeric_summary(df[col])
    if stats["skew"] > 0.5:
        shape = "right-skewed, with a smaller high-end group pulling the average upward"
    elif stats["skew"] < -0.5:
        shape = "left-skewed, with many records concentrated toward higher values"
    else:
        shape = "fairly balanced around the center"

    return (
        f"<b>{col}</b> ranges from {stats['min']:.2f} to {stats['max']:.2f}. "
        f"The mean is {stats['mean']:.2f}, the median is {stats['median']:.2f}, "
        f"and the distribution is {shape}."
    )


def category_insight(df, col):
    counts = df[col].astype(str).value_counts()
    top = counts.index[0]
    pct = counts.iloc[0] / counts.sum() * 100
    return (
        f"<b>{top}</b> is the leading value for <b>{col}</b>, appearing "
        f"{int(counts.iloc[0])} times ({pct:.1f}% of records)."
    )


def relationship_insight(df, x_col, y_col):
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    abs_corr = abs(corr)
    if abs_corr >= 0.65:
        strength = "strong"
    elif abs_corr >= 0.35:
        strength = "moderate"
    else:
        strength = "weak"
    direction = "positive" if corr >= 0 else "negative"
    return (
        f"<b>{x_col}</b> and <b>{y_col}</b> have a {strength} {direction} "
        f"relationship (correlation {corr:.3f})."
    )


def prepare_model_frame(raw_df):
    encoded = raw_df.copy()
    categorical_cols = encoded.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    usable_categorical_cols = []

    for col in categorical_cols:
        unique_count = encoded[col].nunique(dropna=False)
        is_identifier = "name" in col or col.endswith("_id")
        if is_identifier or unique_count > min(50, len(encoded) * 0.35):
            encoded = encoded.drop(columns=[col])
            continue

        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col].astype(str).fillna("missing"))
        usable_categorical_cols.append(col)

    numeric_cols = encoded.select_dtypes(include=np.number).columns.tolist()
    encoded[numeric_cols] = encoded[numeric_cols].replace([np.inf, -np.inf], np.nan)
    encoded[numeric_cols] = encoded[numeric_cols].fillna(encoded[numeric_cols].median())
    return encoded, numeric_cols, usable_categorical_cols


def choose_target(numeric_cols):
    priority = [
        "market_value_million_eur",
        "overall_rating",
        "potential_rating",
        "goals",
        "assists",
        "minutes_played",
        "matches_played",
    ]
    for target in priority:
        if target in numeric_cols:
            return target
    return numeric_cols[0] if numeric_cols else None


def top_numeric_columns(df, numeric_cols, target=None, limit=8):
    preferred = [
        "age",
        "overall_rating",
        "potential_rating",
        "matches_played",
        "goals",
        "assists",
        "minutes_played",
        "market_value_million_eur",
        "contract_years_left",
    ]
    selected = [col for col in preferred if col in numeric_cols]

    if target and target in numeric_cols and target not in selected:
        selected.insert(0, target)

    for col in df[numeric_cols].var().sort_values(ascending=False).index:
        if col not in selected and col != "player_id":
            selected.append(col)

    return selected[:limit]


def add_overview_page(elements, raw_df, encoded_df, numeric_cols, categorical_cols, target, dataset_name, styles):
    missing_cells = int(raw_df.isna().sum().sum())
    duplicate_rows = int(raw_df.duplicated().sum())

    summary_data = [
        ["Metric", "Value"],
        ["Dataset", dataset_name],
        ["Rows", f"{len(raw_df):,}"],
        ["Columns", f"{len(raw_df.columns):,}"],
        ["Numerical columns", f"{len(numeric_cols):,}"],
        ["Categorical columns", f"{len(categorical_cols):,}"],
        ["Missing cells", f"{missing_cells:,}"],
        ["Duplicate rows", f"{duplicate_rows:,}"],
        ["Primary target", target or "N/A"],
    ]

    table = Table(summary_data, colWidths=[2.3 * inch, 3.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f9fafb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    elements.append(Paragraph(f"Dataset Overview: {dataset_name}", styles["heading"]))
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))

    if target and target in encoded_df:
        target_stats = numeric_summary(encoded_df[target])
        target_text = (
            f"The selected prediction target is <b>{target}</b>. Its median is "
            f"{target_stats['median']:.2f}, average is {target_stats['mean']:.2f}, "
            f"and observed range is {target_stats['min']:.2f} to {target_stats['max']:.2f}."
        )
        elements.append(Paragraph(target_text, styles["body"]))


def build_missing_chart(raw_df, dataset_prefix):
    missing_pct = raw_df.isna().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(15)
    if missing_pct.empty:
        return None, "No missing values were found in the dataset."

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(missing_pct.index, missing_pct.values * 100, color="#f97316")
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing values (%)")
    ax.tick_params(axis="x", rotation=45)
    return save_fig(fig, f"{dataset_prefix}_missing_values"), (
        f"The highest missing-value column is <b>{missing_pct.index[0]}</b> at "
        f"{missing_pct.iloc[0] * 100:.1f}%."
    )


def build_correlation_chart(df, numeric_cols, dataset_prefix):
    usable = [col for col in numeric_cols if col != "player_id"]
    if len(usable) < 2:
        return None, "Not enough numerical columns are available for correlation analysis."

    corr = df[usable].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Numerical Correlation Heatmap")

    abs_corr = corr.abs()
    np.fill_diagonal(abs_corr.values, 0)
    strongest_pair = abs_corr.stack().idxmax()
    strongest_value = abs_corr.stack().max()
    insight = (
        f"The strongest numerical relationship is between <b>{strongest_pair[0]}</b> "
        f"and <b>{strongest_pair[1]}</b> with absolute correlation {strongest_value:.3f}."
    )
    return save_fig(fig, f"{dataset_prefix}_correlation_heatmap"), insight


def build_distribution_charts(df, numeric_cols, dataset_prefix):
    charts = []
    for col in top_numeric_columns(df, numeric_cols, limit=9):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(df[col].dropna(), bins=28, color="#2563eb", edgecolor="white")
        ax.axvline(df[col].median(), color="#dc2626", linestyle="--", linewidth=2, label="Median")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Players")
        ax.legend()
        path = save_fig(fig, f"{dataset_prefix}_{col}_distribution")
        charts.append((f"{col} Distribution", distribution_insight(df, col), path))
    return charts


def build_category_charts(raw_df, categorical_cols, dataset_prefix):
    charts = []
    useful_cols = [
        col
        for col in categorical_cols
        if "name" not in col and raw_df[col].nunique(dropna=False) <= min(30, max(2, len(raw_df) * 0.5))
    ]
    for col in useful_cols[:8]:
        counts = raw_df[col].astype(str).value_counts().head(12)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(counts.index, counts.values, color="#16a34a")
        ax.set_title(f"Top {col} Categories")
        ax.set_xlabel(col)
        ax.set_ylabel("Players")
        ax.tick_params(axis="x", rotation=45)
        path = save_fig(fig, f"{dataset_prefix}_{col}_category")
        charts.append((f"Top {col} Categories", category_insight(raw_df, col), path))
    return charts


def build_target_relationship_charts(df, numeric_cols, target, dataset_prefix):
    charts = []
    if not target or target not in numeric_cols:
        return charts

    candidates = [col for col in numeric_cols if col not in (target, "player_id")]
    correlations = df[candidates + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)

    for x_col in correlations.head(8).index:
        plot_df = df[[x_col, target]].dropna()
        if plot_df[x_col].nunique() < 2 or plot_df[target].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(7.4, 5))
        ax.scatter(plot_df[x_col], plot_df[target], s=22, alpha=0.68, color="#7c3aed")
        z = np.polyfit(plot_df[x_col], plot_df[target], 1)
        line_x = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        ax.plot(line_x, z[0] * line_x + z[1], color="#111827", linewidth=1.5)
        ax.set_title(f"{x_col} vs {target}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(target)
        path = save_fig(fig, f"{dataset_prefix}_{x_col}_vs_{target}")
        charts.append((f"{x_col} vs {target}", relationship_insight(df, x_col, target), path))

    return charts


def build_category_target_charts(raw_df, target, dataset_prefix):
    charts = []
    if not target or target not in raw_df.columns:
        return charts

    categorical_cols = [
        col
        for col in raw_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if "name" not in col and raw_df[col].nunique(dropna=False) <= min(30, max(2, len(raw_df) * 0.5))
    ]
    for col in categorical_cols[:6]:
        grouped = raw_df.groupby(col)[target].agg(["mean", "count"]).sort_values("mean", ascending=False)
        grouped = grouped[grouped["count"] >= max(2, len(raw_df) * 0.01)].head(10)
        if grouped.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(grouped.index.astype(str), grouped["mean"], color="#0891b2")
        ax.set_title(f"Average {target} by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(f"Average {target}")
        ax.tick_params(axis="x", rotation=45)
        path = save_fig(fig, f"{dataset_prefix}_avg_{target}_by_{col}")
        insight = (
            f"<b>{grouped.index[0]}</b> has the highest average <b>{target}</b> among "
            f"the displayed <b>{col}</b> groups, at {grouped['mean'].iloc[0]:.2f}."
        )
        charts.append((f"Average {target} by {col}", insight, path))

    return charts


def build_boxplot_charts(raw_df, target, dataset_prefix):
    charts = []
    if not target or target not in raw_df.columns:
        return charts

    categorical_cols = [
        col
        for col in raw_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if "name" not in col and raw_df[col].nunique(dropna=False) <= min(30, max(2, len(raw_df) * 0.5))
    ]
    for col in categorical_cols[:4]:
        top_categories = raw_df[col].astype(str).value_counts().head(8).index
        plot_df = raw_df[raw_df[col].astype(str).isin(top_categories)]
        if plot_df.empty:
            continue

        labels = []
        values = []
        for category, group in plot_df.groupby(col):
            labels.append(str(category))
            values.append(group[target].dropna().values)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(values, labels=labels, patch_artist=True)
        ax.set_title(f"{target} Spread by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.tick_params(axis="x", rotation=45)
        path = save_fig(fig, f"{dataset_prefix}_{target}_boxplot_by_{col}")
        insight = (
            f"This boxplot compares how <b>{target}</b> varies across the most common "
            f"<b>{col}</b> groups, making outliers and median differences easier to spot."
        )
        charts.append((f"{target} Spread by {col}", insight, path))

    return charts


def build_ml_outputs(df, numeric_cols, target, dataset_prefix):
    charts = []
    metrics = {}
    if not target or target not in numeric_cols or len(df) < 20:
        return charts, metrics

    feature_cols = [col for col in numeric_cols if col not in (target, "player_id")]
    if len(feature_cols) < 2:
        return charts, metrics

    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    model_scores = []
    best_model = None
    best_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        model_scores.append((name, rmse, mae, r2))
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    metrics = {
        "best_model": best_name,
        "best_r2": best_r2,
        "scores": model_scores,
    }

    score_df = pd.DataFrame(model_scores, columns=["model", "rmse", "mae", "r2"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(score_df["model"], score_df["r2"], color=["#2563eb", "#16a34a"])
    ax.set_title(f"Model R2 Comparison for {target}")
    ax.set_ylabel("R2 score")
    path = save_fig(fig, f"{dataset_prefix}_{target}_model_comparison")
    insight = (
        f"The best model for predicting <b>{target}</b> is <b>{best_name}</b> "
        f"with R2 {best_r2:.3f}."
    )
    charts.append((f"Model Comparison for {target}", insight, path))

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(importances.index, importances.values, color="#dc2626")
        ax.set_title(f"Top Predictors of {target}")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=45)
        path = save_fig(fig, f"{dataset_prefix}_{target}_feature_importance")
        insight = (
            f"<b>{importances.index[0]}</b> is the strongest model predictor for "
            f"<b>{target}</b>, with importance {importances.iloc[0]:.3f}."
        )
        charts.append((f"Top Predictors of {target}", insight, path))

    return charts, metrics


def build_cluster_outputs(df, numeric_cols, target, dataset_prefix):
    charts = []
    feature_cols = [col for col in numeric_cols if col != "player_id"]
    if len(feature_cols) < 2 or len(df) < 20:
        return charts

    scaled = StandardScaler().fit_transform(df[feature_cols])
    cluster_count = min(5, max(2, len(df) // 80))
    kmeans = KMeans(n_clusters=cluster_count, n_init=15, random_state=42)
    clusters = kmeans.fit_predict(scaled)

    counts = pd.Series(clusters).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(counts.index.astype(str), counts.values, color="#0f766e")
    ax.set_title("Dynamic Player Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Players")
    path = save_fig(fig, f"{dataset_prefix}_cluster_sizes")
    insight = (
        f"KMeans created <b>{cluster_count}</b> player groups. Cluster "
        f"<b>{counts.idxmax()}</b> is largest with {int(counts.max())} players."
    )
    charts.append(("Dynamic Player Cluster Sizes", insight, path))

    if target and target in df.columns:
        cluster_target = pd.DataFrame({"cluster": clusters, target: df[target]}).groupby("cluster")[target].mean()
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.bar(cluster_target.index.astype(str), cluster_target.values, color="#9333ea")
        ax.set_title(f"Average {target} by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"Average {target}")
        path = save_fig(fig, f"{dataset_prefix}_cluster_avg_{target}")
        insight = (
            f"Cluster <b>{cluster_target.idxmax()}</b> has the highest average "
            f"<b>{target}</b> at {cluster_target.max():.2f}."
        )
        charts.append((f"Average {target} by Cluster", insight, path))

    return charts


def build_anomaly_output(df, numeric_cols, target, dataset_prefix):
    charts = []
    feature_cols = [col for col in numeric_cols if col != "player_id"]
    if len(feature_cols) < 2 or len(df) < 20:
        return charts

    scaled = StandardScaler().fit_transform(df[feature_cols])
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(scaled)
    counts = pd.Series(labels).map({1: "Regular", -1: "Unusual"}).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4.7))
    ax.bar(counts.index, counts.values, color=["#2563eb", "#f97316"])
    ax.set_title("Unusual Player Profile Detection")
    ax.set_ylabel("Players")
    path = save_fig(fig, f"{dataset_prefix}_anomaly_detection")
    unusual = int((labels == -1).sum())
    insight = (
        f"Isolation Forest flagged <b>{unusual}</b> statistically unusual player profiles. "
        f"These are useful candidates for deeper scouting, risk review, or valuation checks."
    )
    charts.append(("Unusual Player Profile Detection", insight, path))

    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(7.4, 5))
        normal = df.loc[labels == 1, target]
        anomaly = df.loc[labels == -1, target]
        ax.boxplot([normal, anomaly], labels=["Regular", "Unusual"], patch_artist=True)
        ax.set_title(f"{target} for Regular vs Unusual Profiles")
        ax.set_ylabel(target)
        path = save_fig(fig, f"{dataset_prefix}_{target}_anomaly_boxplot")
        charts.append((f"{target} for Regular vs Unusual Profiles", insight, path))

    return charts


def build_styles():
    getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=HexColor("#111827"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            textColor=HexColor("#2563eb"),
            spaceAfter=26,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontSize=17,
            leading=23,
            textColor=HexColor("#111827"),
            spaceBefore=10,
            spaceAfter=12,
        ),
        "chart_heading": ParagraphStyle(
            "chart_heading",
            fontSize=12.5,
            leading=17,
            textColor=HexColor("#111827"),
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10.5,
            leading=15.5,
            textColor=HexColor("#1f2937"),
            spaceAfter=10,
        ),
    }


def add_chart_to_report(elements, chart, styles):
    elements.append(Paragraph(chart["title"], styles["chart_heading"]))
    elements.append(Image(chart["path"], width=6.25 * inch, height=3.85 * inch))
    elements.append(Paragraph(chart["insight"], styles["body"]))
    elements.append(Spacer(1, 0.16 * inch))


def process_dataset(file_path, elements, styles):
    dataset_name = os.path.basename(file_path)
    raw_df = clean_columns(read_dataset(file_path))
    encoded_df, model_numeric_cols, categorical_cols = prepare_model_frame(raw_df)
    chart_numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    target = choose_target(chart_numeric_cols)
    dataset_prefix = safe_filename(os.path.splitext(dataset_name)[0])

    if not chart_numeric_cols:
        elements.append(Paragraph(f"Dataset skipped: {dataset_name}", styles["heading"]))
        elements.append(Paragraph("No numerical columns were available for analysis.", styles["body"]))
        return []

    add_overview_page(elements, raw_df, encoded_df, chart_numeric_cols, categorical_cols, target, dataset_name, styles)

    charts = []

    missing_path, missing_insight = build_missing_chart(raw_df, dataset_prefix)
    if missing_path:
        add_chart(charts, "Missing Data Profile", missing_insight, missing_path)

    corr_path, corr_insight = build_correlation_chart(raw_df, chart_numeric_cols, dataset_prefix)
    add_chart(charts, "Numerical Correlation Heatmap", corr_insight, corr_path)

    for title, insight, path in build_distribution_charts(raw_df, chart_numeric_cols, dataset_prefix):
        add_chart(charts, title, insight, path)

    for title, insight, path in build_category_charts(raw_df, categorical_cols, dataset_prefix):
        add_chart(charts, title, insight, path)

    for title, insight, path in build_target_relationship_charts(raw_df, chart_numeric_cols, target, dataset_prefix):
        add_chart(charts, title, insight, path)

    for title, insight, path in build_category_target_charts(raw_df, target, dataset_prefix):
        add_chart(charts, title, insight, path)

    for title, insight, path in build_boxplot_charts(raw_df, target, dataset_prefix):
        add_chart(charts, title, insight, path)

    ml_charts, metrics = build_ml_outputs(encoded_df, model_numeric_cols, target, dataset_prefix)
    for title, insight, path in ml_charts:
        add_chart(charts, title, insight, path)

    for title, insight, path in build_cluster_outputs(encoded_df, model_numeric_cols, target, dataset_prefix):
        add_chart(charts, title, insight, path)

    for title, insight, path in build_anomaly_output(encoded_df, model_numeric_cols, target, dataset_prefix):
        add_chart(charts, title, insight, path)

    if target == "market_value_million_eur" and target in raw_df.columns:
        top_players = raw_df.sort_values(target, ascending=False).head(8)
        if "player_name" in top_players.columns:
            names = ", ".join(top_players["player_name"].astype(str).head(5))
            elements.append(
                Paragraph(
                    f"Top market-value players in this dataset include <b>{names}</b>. "
                    f"The highest listed value is <b>{as_money(top_players[target].iloc[0])}</b>.",
                    styles["body"],
                )
            )

    if metrics:
        score_text = "<b>Model performance:</b><br/>"
        for name, rmse, mae, r2 in metrics["scores"]:
            score_text += f"{name}: RMSE {rmse:.3f}, MAE {mae:.3f}, R2 {r2:.3f}<br/>"
        elements.append(Paragraph(score_text, styles["body"]))

    for chart in charts:
        add_chart_to_report(elements, chart, styles)

    elements.append(PageBreak())

    return [
        {
            "dataset": dataset_name,
            "target": target,
            "chart_title": chart["title"],
            "insight": chart["insight"].replace("<b>", "").replace("</b>", ""),
            "chart_path": chart["path"],
        }
        for chart in charts
    ]


def main():
    data_files = find_data_files()
    if not data_files:
        raise ValueError("No CSV or Excel files found in the current folder or Kaggle input.")

    styles = build_styles()
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    elements = [
        Paragraph("FIFA Dynamic Player Analytics Report", styles["title"]),
        Paragraph(
            "Automated dataset-aware reporting with charts, insights, clustering, anomaly detection and ML diagnostics",
            styles["subtitle"],
        ),
        Paragraph(
            "This report is generated dynamically from the available dataset columns. It detects numerical "
            "and categorical fields, selects a prediction target, creates visual diagnostics, and writes "
            "plain-language insights based on the actual data values.",
            styles["body"],
        ),
        PageBreak(),
    ]

    all_insights = []
    for file_path in data_files:
        try:
            all_insights.extend(process_dataset(file_path, elements, styles))
        except Exception as exc:
            elements.append(Paragraph(f"Could not process {os.path.basename(file_path)}", styles["heading"]))
            elements.append(Paragraph(str(exc), styles["body"]))
            elements.append(PageBreak())

    if all_insights:
        pd.DataFrame(all_insights).to_csv(INSIGHT_CSV_PATH, index=False)

    elements.append(Paragraph("Final Summary", styles["heading"]))
    elements.append(
        Paragraph(
            f"Processed <b>{len(data_files)}</b> dataset file(s), generated "
            f"<b>{len(all_insights)}</b> chart-level insights, and saved the chart images under "
            f"<b>{CHART_DIR}</b>.",
            styles["body"],
        )
    )

    doc.build(elements)

    print("Datasets found:", len(data_files))
    print("Insights generated:", len(all_insights))
    print("PDF report generated:", REPORT_PATH)
    print("Insight summary CSV:", INSIGHT_CSV_PATH)
    print("Execution time:", round(time.time() - START_TIME, 2), "seconds")


if __name__ == "__main__":
    main()
