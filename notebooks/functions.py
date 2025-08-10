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


#! Latest
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
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Step 2: Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Step 3: Threshold tuning via PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

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
    plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
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

    # == Calculate sample weight ==
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    # == 4. Train, Threshold Tune, Evaluate, Wandb Log, Log Reg Model ==
    train_eval_logistic_with_threshold(
        X_train=X_train_encoded,
        y_train=y_train,
        X_test=X_test_encoded,
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


#! Spark Final Model Training
def oot_train_test_split(
    initial_df: DataFrame, date_cut_off: str
) -> tuple[DataFrame, DataFrame]:
    """
    Takes in Pyspark Dataframe.
    Conducts out-of-time split according to issue_d column, returning (train_df , test_df)
    """
    initial_df = initial_df.withColumn("issue_d", to_date(col("issue_d"), "yyyy-MM-dd"))
    initial_df = initial_df.orderBy(col("issue_d").asc())
    train_df = initial_df.where(col("issue_d") < to_date(lit(date_cut_off)))
    test_df = initial_df.where(col("issue_d") >= to_date(lit(date_cut_off)))

    return (train_df, test_df)


def add_class_weightage_cols(train_df) -> DataFrame:
    """
    Implement same logic as 'balanced' class in sklearn (give more importance to rare class during training)

    Adds class_weight column to TRAIN_DATASET
    """

    # Count examples in each class
    major_count = train_df.filter(train_df.default_status == 0).count()
    minor_count = train_df.filter(train_df.default_status == 1).count()
    total_count = train_df.count()

    # Calculate weights (inverse frequency)
    weight_for_0 = total_count / (2 * major_count)
    weight_for_1 = total_count / (2 * minor_count)

    # Add a column for sample weights
    train_df = train_df.withColumn(
        "class_weight_col",
        F.when(train_df.default_status == 0, weight_for_0).otherwise(weight_for_1),
    )

    return train_df


def build_one_hot_encoding_pipeline(df):
    """
    Turns all categories (<string_type>) into one_hot_encoded columns and returns Spark DataFrame
    """
    cat_features = [
        feature.name
        for feature in df.schema.fields
        if isinstance(feature.dataType, StringType)
    ]

    # Create stages for Pipeline: Indexers + Encoders
    stages = []
    for cat_feature in cat_features:
        # StringIndexer stage
        if df.select(cat_feature).distinct().count() > 1:
            indexer = StringIndexer(
                inputCol=cat_feature, outputCol=f"{cat_feature}_idx"
            )
            # OneHotEncoder stage (uses indexer's output)
            encoder = OneHotEncoder(
                inputCol=f"{cat_feature}_idx",
                outputCol=f"{cat_feature}_one_hot_encoded",
                dropLast=True,
            )
            stages.extend([indexer, encoder])

    return Pipeline(stages=stages)


# def calculate_vif_spark(df, features, threshold=5.0, sample_frac=0.05, seed=42):
#     """
#     For each feature of the df, calculate the Variance Inflation Factor (VIF) to check for multicollinearity.
#     """
#     work_df = (
#         df.select(features)  # keep only numeric columns
#         .sample(fraction=sample_frac, seed=seed)
#         .persist(
#             StorageLevel.MEMORY_AND_DISK
#         )  # store the sampled in ram, spill to disk if needed (impt since i need to assemble vectors of other features, for each given feature) -> constant ref to this df
#     )
#     _ = work_df.count()  # materialize cache

#     vif_scores = []

#     # --- 2)  Loop through features to compute VIF -------------
#     for idx, feature in enumerate(features, start=1):
#         other = [c for c in features if c != feature]

#         # Assemble other features into a single vector
#         assembler = VectorAssembler(inputCols=other, outputCol="features_vec")
#         temp = assembler.transform(work_df).select("features_vec", feature)

#         # Regress feature ~ other features  → get R²
#         lr = LinearRegression(
#             featuresCol="features_vec", labelCol=feature, regParam=0.001
#         )
#         r2 = lr.fit(temp).summary.r2

#         vif = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
#         vif_scores.append((feature, vif))

#     # --- 3)  Split keep vs. drop ------------------------------
#     keep_cols = [(f, v) for f, v in vif_scores if v <= threshold]
#     drop_cols = [(f, v) for f, v in vif_scores if v > threshold]

#     return (keep_cols, drop_cols)


# # Log and Evaluate Temporary Results (Each data preprocessing step)
# def run_model_checkpoint(df, name, date_cutoff, sample_proportion, model_type, group):
#     """
#     Takes in Spark Dataframe & conducts necessary steps to train a Logistic Regression model quickly (Aim: Evaluate model performance quickly to
#     figure out decrease in model performance)

#     Input Param:
#     - name: logistic_regression_base_features / similar naming after preprocessing steps
#     - date_cutoff: 2018-01-01 by default
#     - model_type: 'logistic_regression' by default, but can be xgboost, lightgbm and others ...
#     - group: pd_model_bulding_1 for the current notebook, but can be pd_model_building_2, pd_model_building_3, etc
#     """

#     # 1. Start a run (preprocessing step) ... (just set a new name when i run)
#     wandb.init(
#         entity="wlunlun1212-singapore-management-university",
#         project="Credit Risk Modeling",
#         name=name,  #! change to logistic_regression_base_features b4 rest of the preprocessing steps
#         group=group,
#     )

#     sampled_df = sample_and_order(df, sample_proportion)
#     train_df, test_df = oot_train_test_split(sampled_df, date_cutoff)
#     train_weighted_df = add_class_weightage_cols(train_df)
#     test_weighted_df = test_df.withColumn(
#         "class_weight_col", lit(1.0)
#     )  #! Ensure test_df has class_weight_col for consistency, even if it is not used in evaluation (weight column is not read in evaluation)

#     # 2. Ensure train_df can be fed into LR_Model with 'features' vector column, 'class_weight_col' weightCol, 'default_status' label column
#     excluded_cols = [
#         "id",
#         "issue_d",
#         "default_status",
#         "class_weight_col",
#         "earliest_cr_line",
#     ]  # exclude date columns, target columns, id for features

#     # 2a. One-Hot Encode both train & test datasets if there are categorical features, b4 fitting into Vector Assembler: one-hot-encoder should only see train data, and not test data
#     cat_features = [
#         feature.name
#         for feature in train_df.schema.fields
#         if isinstance(feature.dataType, StringType)
#     ]

#     # Ensure that the categorical features are not empty, otherwise skip one-hot encoding
#     if len(cat_features) > 0:
#         ohe_pipeline = build_one_hot_encoding_pipeline(
#             train_weighted_df
#         )  # create pipeline object, which has StringIndexer + OneHotEncoder stages, i.e.
#         # Pipeline(stages=[
#         # StringIndexer(inputCol='grade',           outputCol='grade_idx',           handleInvalid='keep'),
#         # OneHotEncoder(inputCol='grade_idx',       outputCol='grade_ohe',           dropLast=True),
#         # StringIndexer(inputCol='home_ownership',  outputCol='home_ownership_idx',  handleInvalid='keep'),
#         # OneHotEncoder(inputCol='home_ownership_idx',
#         #             outputCol='home_ownership_ohe',
#         #             dropLast=True)
#         # ])
#         ohe_model = ohe_pipeline.fit(
#             train_weighted_df
#         )  # schedule of Stringindexers + Onehotencoders
#         train_encoded_df = ohe_model.transform(train_weighted_df).cache()
#         test_encoded_df = ohe_model.transform(test_weighted_df)

#         idx_cols = [f"{cat}_idx" for cat in cat_features]
#         train_encoded_df = train_encoded_df.drop(
#             *idx_cols
#         )  # drop index columns created by StringIndexer
#         test_encoded_df = test_encoded_df.drop(
#             *idx_cols
#         )  # drop index columns created by StringIndexer

#     else:
#         train_encoded_df = train_weighted_df.cache()
#         test_encoded_df = test_weighted_df

#     feature_cols = [
#         f.name
#         for f in train_encoded_df.schema.fields
#         if (
#             (
#                 isinstance(f.dataType, NumericType) or isinstance(f.dataType, VectorUDT)
#             )  # VectorUDT datatype of one-hot-encoded column
#             and f.name not in excluded_cols
#             and not f.name.endswith(
#                 "_idx"
#             )  # skip index columns created by StringIndexer
#         )
#     ]

#     vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#     # 3. Feed Vectors, Label, Class into Logistic Regression Model (Excludes idx from features ... )
#     lr = LogisticRegression(
#         featuresCol="features",
#         labelCol="default_status",
#         weightCol="class_weight_col",
#         regParam=0.01,  #! Lower values mean potential to overfit more on training data ...
#     )

#     pipeline = Pipeline(stages=[vector_assembler, lr])
#     trained_lr_model = pipeline.fit(train_encoded_df)
#     train_encoded_df.unpersist()  # release memory

#     # 4. Predict
#     preds = trained_lr_model.transform(test_encoded_df).cache()

#     # Example of preds: (contains original columns + model_output )
#     # income | dti | ... | features | prediction | probability |	rawPrediction
#     # 1000 | 0.99 |  [0.1, 1.5, 2.4, ...]	| 1.0	| [0.21, 0.79]	| [-1.32, 1.32]

#     # 5. Gini Coefficient
#     evaluator_bin = BinaryClassificationEvaluator(
#         labelCol="default_status",
#         rawPredictionCol="rawPrediction",
#         metricName="areaUnderROC",
#     )
#     auc = evaluator_bin.evaluate(
#         preds
#     )  # materialises cache of 'preds' -> any subsequent read of 'preds' is from memory
#     gini = 2 * auc - 1

#     # 6. Obtain confusion matrix related metrics for model performance logging
#     evaluator_multi = MulticlassClassificationEvaluator(
#         labelCol="default_status", predictionCol="prediction"
#     )
#     f1 = evaluator_multi.setMetricName("f1").evaluate(preds)
#     precision = evaluator_multi.setMetricName("weightedPrecision").evaluate(preds)
#     recall = evaluator_multi.setMetricName("weightedRecall").evaluate(preds)
#     accuracy = evaluator_multi.setMetricName("accuracy").evaluate(preds)

#     # 7. Get parameters for Wandb Confusion Matrix

#     rows = preds.select("default_status", "prediction").collect()
#     y_actual = [row["default_status"] for row in rows]
#     y_pred = [row["prediction"] for row in rows]

#     # 8. Log to W&B
#     wandb.log(
#         {
#             "Model Type": model_type,
#             "Gini": gini,
#       no..       "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1,
#             "Confusion Matrix": wandb.plot.confusion_matrix(
#                 y_true=y_actual, preds=y_pred, class_names=["Non-Default", "Default"]
#             ),
#         }
#     )

#     wandb.finish()


# Check multicollinearity across notebooks
# def calculate_vif(df, features, threshold=5.0, sample_frac=0.05, seed=42):
#     """
#     For each feature of the df, calculate the Variance Inflation Factor (VIF) to check for multicollinearity.
#     """
#     work_df = (
#         df.select(features)  # keep only numeric columns
#         .sample(fraction=sample_frac, seed=seed)
#         .cache()
#     )
#     _ = work_df.count()  # materialize cache

#     vif_scores = []

#     # --- 2)  Loop through features to compute VIF -------------
#     for idx, feature in enumerate(features, start=1):
#         other = [c for c in features if c != feature]

#         # Assemble other features into a single vector
#         assembler = VectorAssembler(inputCols=other, outputCol="features_vec")
#         temp = assembler.transform(df).select("features_vec", feature)

#         # Regress feature ~ other features  → get R²
#         lr = LinearRegression(
#             featuresCol="features_vec", labelCol=feature, regParam=0.001
#         )
#         r2 = lr.fit(temp).summary.r2

#         vif = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
#         vif_scores.append((feature, vif))

#     # --- 3)  Split keep vs. drop ------------------------------
#     keep_cols = [(f, v) for f, v in vif_scores if v <= threshold]
#     drop_cols = [(f, v) for f, v in vif_scores if v > threshold]

#     work_df.unpersist()

#     return (keep_cols, drop_cols)


# import statsmodels

# print(statsmodels.__version__)

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.preprocessing import StandardScaler
