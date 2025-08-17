# =========================
# Standard library
# =========================
import os
import string
from datetime import date
from random import sample  # if you really need it

# =========================
# Third-party: Environment & Logging
# =========================
from dotenv import load_dotenv
import wandb

# =========================
# Third-party: PySpark
# =========================
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import (
    col,
    when,
    count,
    desc,
    isnan,
    isnull,
    lit,
    length,
    trim,
    lower,
    upper,
    to_date,
    concat_ws,
    regexp_extract,
    sum,
    unix_timestamp,
    from_unixtime,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    DateType,
    NumericType,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import VectorUDT  # if you actually use it

# =========================
# Third-party: Data (Pandas / NumPy)
# =========================
import numpy as np
import pandas as pd

# =========================
# Third-party: Plotting
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Third-party: Scikit-learn
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# =========================
# Third-party: Statsmodels (VIF)
# =========================
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =========================
# Project setup
# =========================
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


# == Outlier Handling ===
def winsorise_col_percentile(
    df: DataFrame,
    col_name: str,
    lower_pct: float = 0.005,  # e.g., 0.5% quantile
    upper_pct: float = 0.995,  # e.g., 99.5% quantile
    rel_err: float = 1e-3,
) -> DataFrame:
    """
    Winsorises a column by capping values below/above percentile thresholds.

    Args:
        df (DataFrame): Input Spark DataFrame
        col_name (str): Column to winsorise
        lower_pct (float): Lower quantile (default 0.005 = 0.5%)
        upper_pct (float): Upper quantile (default 0.995 = 99.5%)
        rel_err (float): Relative error for approxQuantile

    Returns:
        DataFrame: With winsorised column
    """
    print(
        f"✅ Winsorising column: {col_name} using percentiles {lower_pct*100:.2f}%–{upper_pct*100:.2f}% ..."
    )

    # Compute quantiles
    quantiles = df.approxQuantile(col_name, [lower_pct, upper_pct], rel_err)
    if len(quantiles) == 2:
        q_low, q_high = quantiles
        df = df.withColumn(
            col_name,
            when(col(col_name) < q_low, q_low)
            .when(col(col_name) > q_high, q_high)
            .otherwise(col(col_name)),
        )
        return df
    else:
        print(
            f"⚠️ Could not compute quantiles for column '{col_name}'. Is the column empty or all NaN?"
        )
    return df

    #


def winsorise_col(df, col_name, operator: str, condition_val, final_val):
    """
    Winsorises a column by replacing values above a certain condition with a final value.

    Args:
        df (DataFrame): The input DataFrame.
        col_name (str): The name of the column to winsorise.
        condition_val (float): The value above which to replace with final_val (cut-off)
        final_val (float): The value to replace with.

    Returns:
        DataFrame: The DataFrame with the winsorised column.
    """
    print("✅ Winsorising column:", col_name, "...")

    if operator == "<":
        return df.withColumn(
            col_name,
            when(col(col_name) < condition_val, final_val).otherwise(col(col_name)),
        )

    elif operator == ">":
        return df.withColumn(
            col_name,
            when(col(col_name) > condition_val, final_val).otherwise(col(col_name)),
        )


def retain_rows(
    df: DataFrame, col_name: str, condition_val: float, operator: str
) -> DataFrame:
    """
    Retains rows in the DataFrame where the specified column meets a condition.

    Returns:
        DataFrame: The DataFrame with the specified rows dropped.
    """

    if operator == "<=":
        return df.filter(col(col_name) <= condition_val)

    elif operator == "<":
        return df.filter(col(col_name) < condition_val)

    elif operator == ">":
        return df.filter(col(col_name) > condition_val)

    elif operator == ">=":
        return df.filter(col(col_name) >= condition_val)

    else:
        raise ValueError("Operator must be '>=' or '<='")


def compute_outlier_pct(df, col_name, lower_pct=0.25, upper_pct=0.75):
    """Computes pct of outliers per column based on IQR method"""

    # 1. Compute percentile bounds
    quantiles = df.approxQuantile(col_name, [lower_pct, upper_pct], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1

    # 2. Obtain lower and upper bound, any data points outside of this are seen as outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    total_rows = df.count()

    return round(
        df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
        / total_rows
        * 100,
        2,
    )


def display_distributions(df):
    """Takes in Spark Dataframe. Samples it and display distribution for skewness checking"""
    # 1. Select numerical columns
    numeric_cols = sorted(
        [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, NumericType)
        ]
    )

    # 2. Sample small portion of data (e.g., 5%) and convert to pandas
    sample_df = df.select(numeric_cols).sample(fraction=0.1, seed=42)
    sample_pdf = sample_df.toPandas()

    # 3. Plot histograms as subplots
    n_cols = 3  # Number of plots per row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col_name in enumerate(numeric_cols):
        axes[i].hist(sample_pdf[col_name].dropna(), bins=50, color="skyblue")
        axes[i].set_title(col_name, fontsize=10)
        axes[i].tick_params(axis="x", rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def inspect_outliers(
    df,
    columns: list,
    sample_size: int = 5,
    method: str = "iqr",  # "iqr" or "percentile"
    lower_pct: float = 0.01,  # used if method="percentile"
    upper_pct: float = 0.99,  # used if method="percentile"
):
    total_count = df.count()

    for col_name in columns:
        try:
            print(f"\n📊 Inspecting Outliers for Column: `{col_name}`")

            if method == "iqr":
                q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                print(f"IQR Method | Q1 = {q1}, Q3 = {q3}, IQR = {iqr}")
                print(f"Lower Bound = {lower}, Upper Bound = {upper}")

            elif method == "percentile":
                # Get both percentile bounds
                lower, upper = df.approxQuantile(col_name, [lower_pct, upper_pct], 0.01)

                # NEW: also inspect the max value
                max_val = df.agg(F.max(df[col_name]).alias("max")).first()["max"]
                print(
                    f"Percentile Method | Lower (p{int(lower_pct*100)}) = {lower}, "
                    f"Upper (p{int(upper_pct*100)}) = {upper}"
                )
                print(f"🔝 Max({col_name}) = {max_val}")

                # OPTIONAL: show a few rows at the exact max
                df.filter(df[col_name] == max_val).select(col_name).show(
                    min(sample_size, 5)
                )

            else:
                print(f"❌ Unknown method `{method}`. Skipping column `{col_name}`.")
                continue

            # Count outliers
            outlier_count = df.filter(
                (df[col_name] < lower) | (df[col_name] > upper)
            ).count()
            outlier_pct = round(outlier_count / total_count * 100, 2)
            print(f"Outlier Count: {outlier_count} ({outlier_pct}%)")

            # Sample outliers top and bottom
            print(f"🔼 Top Outliers (>{upper}):")
            df.filter(df[col_name] > upper).select(col_name).orderBy(
                df[col_name].desc()
            ).show(sample_size)

            print(f"🔽 Bottom Outliers (<{lower}):")
            df.filter(df[col_name] < lower).select(col_name).orderBy(
                df[col_name].asc()
            ).show(sample_size)

        except Exception as e:
            print(f"❌ Could not process column `{col_name}`: {str(e)}")


# == PD Modeling ==


def train_eval_logistic_with_threshold(
    X_train,
    y_train,
    X_test,
    y_test,
    labels=["Non-Default", "Default"],
    sample_weight=None,
    model_type=None,
):
    """Train Logistic Regression Model with threshold, outputs evaluation metrics and logs it to Wandb.
    Focuses on Default Class metrics, rather than weighted"""
    # Step 1: Train.
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, penalty="l2")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Step 2: Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Step 3: Threshold tuning via PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # precision/recall are length N+1, thresholds is length N → align by skipping index 0
    f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-12)

    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    print(f"\n✅ Best F1 Score = {best_f1:.4f} at threshold = {best_thresh:.2f}")

    # Step 4: Final prediction using optimal threshold
    y_pred_opt = (y_proba >= best_thresh).astype(int)

    # Step 5: Confusion matrix
    cm = confusion_matrix(y_test, y_pred_opt)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Optimized Confusion Matrix (F1)")
    plt.tight_layout()
    cm_fig = plt.gcf()
    plt.close()

    # Step 5: Threshold F1 Score ==
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, f1_scores, label="F1 Score")

    plt.axvline(
        best_thresh,
        color="red",
        linestyle="--",
        label=f"Best Threshold: {best_thresh:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold")
    plt.legend()
    f1_fig = plt.gcf()
    plt.close()

    # Step 6: Print metrics table
    accuracy = accuracy_score(y_test, y_pred_opt)
    precision_final = precision_score(y_test, y_pred_opt)
    recall_final = recall_score(y_test, y_pred_opt)
    f1_final = f1_score(y_test, y_pred_opt)
    auc = roc_auc_score(y_test, y_proba)
    gini = 2 * auc - 1

    # == Classification Report ==
    print("\n📄 Classification Report (Optimized Threshold):")
    print(classification_report(y_test, y_pred_opt, target_names=labels, digits=4))

    report_dict = classification_report(
        y_test, y_pred_opt, target_names=labels, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T.round(4)
    report_md = report_df.to_markdown()

    wandb.log(
        {
            "Model Type": model_type,
            "Best Threshold": best_thresh,
            "Gini": gini,
            "Accuracy": accuracy,
            "Precision": precision_final,
            "Recall": recall_final,
            "F1 Score": f1_final,
            "Confusion Matrix (sns)": wandb.Image(cm_fig),
            "F1 vs Threshold": wandb.Image(f1_fig),
            "Classification Report (Markdown)": wandb.Html(f"<pre>{report_md}</pre>"),
        }
    )
    wandb.finish()


def run_model_checkpoint(
    train_pdf: DataFrame,
    test_pdf: DataFrame,
    run_name: str,
    model_type: str,
    run_group: str,
):
    """
    Before executing this function, ensure a sampled_df is used to generate train_pdf, test_pdf.

    Takes in train_pdf, test_pdf (Pandas form) and sort them by `issue_d` ascendingly. Conducts necessary steps
    to train Logistic Regression Model (PD). Logs Accuracy, Gini, F1 Score, Recall, Precision,
    Confusion Matrix into Wandb for quick model evaluation at each data preprocessing stage.

    Input Param:
    - train_pdf / test_pdf : Pandas Dataframe
    - name: logistic_regression_base_features / similar naming after preprocessing steps
    - date_cutoff: 2018-01-01 by default
    - model_type: 'logistic_regression' by default, but can be xgboost, lightgbm and others ...
    - group: pd_model_bulding_1 for the current notebook, but can be pd_model_building_2, pd_model_building_3, etc
    """
    # == 1. Start a run (preprocessing step) ... (just set a new name when i run) ==
    wandb.init(
        entity="wlunlun1212-singapore-management-university",
        project="Credit Risk Modeling",
        name=run_name,  #! change to logistic_regression_base_features b4 rest of the preprocessing steps
        group=run_group,
    )

    #! == 2. Undersample majority class on train dataset -> Class imbalance handling == (Changed to Class
    #! Weighting to fit Spark)
    # majority = train_pdf[train_pdf["default_status"] == 0]
    # minority = train_pdf[train_pdf["default_status"] == 1]
    # majority_downsampled = majority.sample(n=len(minority), random_state=42)
    # train_pdf_balanced = pd.concat([majority_downsampled, minority]).sample(
    #     frac=1, random_state=42
    # )

    # == 3. Prepare features ==
    X_train = train_pdf.drop(columns=["id", "issue_d", "default_status"])
    y_train = train_pdf["default_status"]

    X_test = test_pdf.drop(columns=["id", "issue_d", "default_status"])
    y_test = test_pdf["default_status"]

    # == Dummy Encoding ==
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)

    # Align columns (test might be missing some)
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

    # == Standarization ==
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # == Calculate sample weight ==
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    # == 4. Train, Threshold Tune, Evaluate, Wandb Log, Log Reg Model ==
    train_eval_logistic_with_threshold(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        sample_weight=sample_w,
        model_type="Logistic Regression",
    )


def sample_split_order(
    initial_df,
    sample_frac,
    cut_off_date,
    date_col,
):
    """
    Takes a Spark Dataframe and samples it. Sorts Pandas Dataframe (sampled) accordingly.
    Splits sampled data into train and test dataframes.

    Parameters
    ----------
    initial_df: Dataframe
            Spark Dataframe

    sample_frac: float
            How much to sample from Big Data

    cut_off_date: str
            Pandas (pd.date_time())

    date_col: str
            'issue_d' column
    """
    sampled_pdf = initial_df.sample(
        withReplacement=False, fraction=sample_frac, seed=42
    ).toPandas()
    sampled_pdf.sort_values(by=date_col, inplace=True)

    sampled_pdf[date_col] = pd.to_datetime(sampled_pdf[date_col], errors="coerce")

    train_pdf = sampled_pdf[sampled_pdf[date_col] <= cut_off_date]
    test_pdf = sampled_pdf[sampled_pdf[date_col] > cut_off_date]

    print(f"train_pdf has {train_pdf.shape[0]} rows, {train_pdf.shape[1]} columns")
    print(f"test_pdf has {test_pdf.shape[0]} rows, {test_pdf.shape[1]} columns")

    return train_pdf, test_pdf


def calculate_vif_pandas(df, features, threshold=5.0):
    """
    Calculates VIF scores for each feature in a Pandas DataFrame.

    Parameters:
    - df (DataFrame): Input Pandas DataFrame
    - features (list of str): List of feature names to test
    - threshold (float): VIF threshold to drop for multicollinearity

    Returns:
    - keep_cols: [(feature, VIF)]
    - drop_cols: [(feature, VIF)]
    """

    # 1. Standardize features to remove scale effects
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # 2. Add constant for intercept
    X_const = add_constant(X_scaled)

    # 3. Calculate VIF for each feature (skip the intercept)
    vif_scores = []
    for i in range(1, X_const.shape[1]):  # start at 1 to skip 'const'
        vif = variance_inflation_factor(X_const.values, i)
        vif_scores.append((features[i - 1], vif))

    # 4. Split into keep/drop lists
    keep_cols = [(f, v) for f, v in vif_scores if v <= threshold]
    drop_cols = [(f, v) for f, v in vif_scores if v > threshold]

    return keep_cols, drop_cols
