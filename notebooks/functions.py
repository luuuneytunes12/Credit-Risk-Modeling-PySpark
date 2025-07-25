# Data manipulation
from pyspark.sql import functions as F
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

from pyspark.ml.linalg import VectorUDT


from pyspark.sql import DataFrame
from datetime import date


# Machine Learning part
from optbinning import BinningProcess
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd

# Third-party packages
import wandb
import os
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

from pyspark.ml.regression import LinearRegression
from pyspark import StorageLevel


# 1. To train model at each data preprocessing step


def sample_and_order(
    initial_df: DataFrame, sample_frac: float, date_col: str = "issue_d"  # Spark df
) -> DataFrame:
    """
    Returns sampled data, with data ordered by the specified date column.
    """
    df = initial_df.sample(withReplacement=False, fraction=sample_frac, seed=42).cache()
    df = df.withColumn(date_col, to_date(col(date_col), "yyyy-MM-dd"))
    df = df.orderBy(col(date_col).asc())
    return df


def oot_train_test_split(
    initial_df: DataFrame, date_cut_off: str
) -> tuple[DataFrame, DataFrame]:
    """
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


# Log and Evaluate Temporary Results (Each data preprocessing step)
def run_model_checkpoint(df, name, date_cutoff, sample_proportion, model_type, group):
    """
    Takes in Spark Dataframe & conducts necessary steps to train a Logistic Regression model quickly (Aim: Evaluate model performance quickly to
    figure out decrease in model performance)

    Input Param:
    - name: logistic_regression_base_features / similar naming after preprocessing steps
    - date_cutoff: 2018-01-01 by default
    - model_type: 'logistic_regression' by default, but can be xgboost, lightgbm and others ...
    - group: pd_model_bulding_1 for the current notebook, but can be pd_model_building_2, pd_model_building_3, etc
    """

    # 1. Start a run (preprocessing step) ... (just set a new name when i run)
    wandb.init(
        entity="wlunlun1212-singapore-management-university",
        project="Credit Risk Modeling",
        name=name,  #! change to logistic_regression_base_features b4 rest of the preprocessing steps
        group=group,
    )

    sampled_df = sample_and_order(df, sample_proportion)
    train_df, test_df = oot_train_test_split(sampled_df, date_cutoff)
    train_weighted_df = add_class_weightage_cols(train_df)
    test_weighted_df = test_df.withColumn(
        "class_weight_col", lit(1.0)
    )  #! Ensure test_df has class_weight_col for consistency, even if it is not used in evaluation (weight column is not read in evaluation)

    # 2. Ensure train_df can be fed into LR_Model with 'features' vector column, 'class_weight_col' weightCol, 'default_status' label column
    excluded_cols = [
        "id",
        "issue_d",
        "default_status",
        "class_weight_col",
        "earliest_cr_line",
    ]  # exclude date columns, target columns, id for features

    # 2a. One-Hot Encode both train & test datasets if there are categorical features, b4 fitting into Vector Assembler: one-hot-encoder should only see train data, and not test data
    cat_features = [
        feature.name
        for feature in train_df.schema.fields
        if isinstance(feature.dataType, StringType)
    ]

    # Ensure that the categorical features are not empty, otherwise skip one-hot encoding
    if len(cat_features) > 0:
        ohe_pipeline = build_one_hot_encoding_pipeline(
            train_weighted_df
        )  # create pipeline object, which has StringIndexer + OneHotEncoder stages, i.e.
        # Pipeline(stages=[
        # StringIndexer(inputCol='grade',           outputCol='grade_idx',           handleInvalid='keep'),
        # OneHotEncoder(inputCol='grade_idx',       outputCol='grade_ohe',           dropLast=True),
        # StringIndexer(inputCol='home_ownership',  outputCol='home_ownership_idx',  handleInvalid='keep'),
        # OneHotEncoder(inputCol='home_ownership_idx',
        #             outputCol='home_ownership_ohe',
        #             dropLast=True)
        # ])
        ohe_model = ohe_pipeline.fit(
            train_weighted_df
        )  # schedule of Stringindexers + Onehotencoders
        train_encoded_df = ohe_model.transform(train_weighted_df).cache()
        test_encoded_df = ohe_model.transform(test_weighted_df)

        idx_cols = [f"{cat}_idx" for cat in cat_features]
        train_encoded_df = train_encoded_df.drop(
            *idx_cols
        )  # drop index columns created by StringIndexer
        test_encoded_df = test_encoded_df.drop(
            *idx_cols
        )  # drop index columns created by StringIndexer

    else:
        train_encoded_df = train_weighted_df.cache()
        test_encoded_df = test_weighted_df

    feature_cols = [
        f.name
        for f in train_encoded_df.schema.fields
        if (
            (
                isinstance(f.dataType, NumericType) or isinstance(f.dataType, VectorUDT)
            )  # VectorUDT datatype of one-hot-encoded column
            and f.name not in excluded_cols
            and not f.name.endswith(
                "_idx"
            )  # skip index columns created by StringIndexer
        )
    ]

    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # 3. Feed Vectors, Label, Class into Logistic Regression Model (Excludes idx from features ... )
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="default_status",
        weightCol="class_weight_col",
        regParam=0.01,  #! Lower values mean potential to overfit more on training data ...
    )

    pipeline = Pipeline(stages=[vector_assembler, lr])
    trained_lr_model = pipeline.fit(train_encoded_df)
    train_encoded_df.unpersist()  # release memory

    # 4. Predict
    preds = trained_lr_model.transform(test_encoded_df).cache()

    # Example of preds: (contains original columns + model_output )
    # income | dti | ... | features | prediction | probability |	rawPrediction
    # 1000 | 0.99 |  [0.1, 1.5, 2.4, ...]	| 1.0	| [0.21, 0.79]	| [-1.32, 1.32]

    # 5. Gini Coefficient
    evaluator_bin = BinaryClassificationEvaluator(
        labelCol="default_status",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator_bin.evaluate(
        preds
    )  # materialises cache of 'preds' -> any subsequent read of 'preds' is from memory
    gini = 2 * auc - 1

    # 6. Obtain confusion matrix related metrics for model performance logging
    evaluator_multi = MulticlassClassificationEvaluator(
        labelCol="default_status", predictionCol="prediction"
    )
    f1 = evaluator_multi.setMetricName("f1").evaluate(preds)
    precision = evaluator_multi.setMetricName("weightedPrecision").evaluate(preds)
    recall = evaluator_multi.setMetricName("weightedRecall").evaluate(preds)
    accuracy = evaluator_multi.setMetricName("accuracy").evaluate(preds)

    # 7. Get parameters for Wandb Confusion Matrix

    rows = preds.select("default_status", "prediction").collect()
    y_actual = [row["default_status"] for row in rows]
    y_pred = [row["prediction"] for row in rows]

    # 8. Log to W&B
    wandb.log(
        {
            "Model Type": model_type,
            "Gini": gini,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=y_actual, preds=y_pred, class_names=["Non-Default", "Default"]
            ),
        }
    )

    wandb.finish()


# Check multicollinearity across notebooks
def calculate_vif(df, features, threshold=5.0, sample_frac=0.05, seed=42):
    """
    For each feature of the df, calculate the Variance Inflation Factor (VIF) to check for multicollinearity.
    """
    work_df = (
        df.select(features)  # keep only numeric columns
        .sample(fraction=sample_frac, seed=seed)
        .persist(
            StorageLevel.MEMORY_AND_DISK
        )  # store the sampled in ram, spill to disk if needed (impt since i need to assemble vectors of other features, for each given feature) -> constant ref to this df
    )
    _ = work_df.count()  # materialize cache

    vif_scores = []

    # --- 2)  Loop through features to compute VIF -------------
    for idx, feature in enumerate(features, start=1):
        other = [c for c in features if c != feature]

        # Assemble other features into a single vector
        assembler = VectorAssembler(inputCols=other, outputCol="features_vec")
        temp = assembler.transform(work_df).select("features_vec", feature)

        # Regress feature ~ other features  → get R²
        lr = LinearRegression(
            featuresCol="features_vec", labelCol=feature, regParam=0.001
        )
        r2 = lr.fit(temp).summary.r2

        vif = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
        vif_scores.append((feature, vif))

    # --- 3)  Split keep vs. drop ------------------------------
    keep_cols = [(f, v) for f, v in vif_scores if v <= threshold]
    drop_cols = [(f, v) for f, v in vif_scores if v > threshold]

    return (keep_cols, drop_cols)
