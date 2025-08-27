## 1. Project Description

<br>
<img src="./images/Credit-Risk-Modelling-1.webp" width='100%'>

In this project, I will be exploring the world of credit risk modeling in banks. To simulate this, I will be using a dataset provided by Lending Club, a peer-to-peer lending company. Though slightly different from the business of traditional banks loan consumers, this project would be a good stepping stone for me to learn the fundamentals of credit risk modeling.

Specifically, i will be building ML models to predict PD (Probability of Default) via regression and tree machine learning models, since they are widely used in the credit risk domain, to improve credit policies of banks etc.

I will be incorporating the following ideas in the data workflow for this project, such as implementing a Medallion Architecture (to simulate data engineering workflows), utilising Pyspark for big data processing, building both baseline and challenger models, and also MLOps operations, such as logging experiment run.

I will be utilising the Crisp-DM Data Science Framework seen below to maintain data integrity, especially given the strict financial regulations of Basel III.

<p style="text-align:center;">
<img src="./images/1200px-CRISP-DM_Process_Diagram.png" width="50%">
</p>

## 2. Business Understanding & Problem

##### 2.1 Business Understanding

- **Lending Club:** Peer-Peer lending platform / Digital Bank, allowing borrowers to seek personal loans from lenders, without the need for traditional banks.

- **Lending Club Revenue Source**: Upon borrower receiving loan, Lending Club charges a loan processing fee / commission fee (Major source of income: Before 2021)

- **Lending Club Credit Risk Workflow**:

  1. Borrowers apply loan on Lending Club marketplace, providing personal financial information
  2. Lending Club use **credit risk modeling** to assign a risk grade, connecting them to interested lenders
  3. Lending Club earn **commission fees** from connecting borrowers and lenders, while lenders earn **interest** from loans

- **Why care about Credit Risk Modeling?**
  - **Too strict?**: Lose good borrowers and opportunities to earn commission fees, leading to **revenue loss** 💸
  - **Too lenient**?: Increased number of defaults lead to loss of trust of investors in Lending Club, leading to **revenue loss** 💸

##### 2.2 Business Problem

- Lending Club wants to **reduce its financial losses** by capturing default loans more effectively. It sees the need to build **highly accurate credit risk models**, reduce its risk appetite.

- **Task:** As a junior data scientist, I will need to **help Lending Club build credit risk models, to predict the probability of default of a loan. s**

##### 2.3 Project Objectives

1. Build a **baseline Probability of Default (PD) model** that effectively identifies high-risk loans, serving as a reliable benchmark for comparison.

2. Develop a **challenger PD model** to improve upon the baseline’s predictive performance, enabling continuous enhancement in risky loan detection

3. Apply **comprehensive model evaluation** techniques that meet the stringent requirements of Basel III financial regulations

## 3. Project Directory Guideline

- **data/**: Output by Pyspark Medallion Architecture
- **images/**: Contains images needed for storytelling and visualisations
- **notebooks/**: Contains Jupyter notebooks for data preprocessing, model building and model evaluation
- **sandbox/**: For data exploration tasks (e.g. identify issues with data for subsequent data preocessing)

## 4. Tech Stack

- **PySpark** Big Data Processing & Engineering / Medallion Architecture
- **Pandas / Numpy**: Data Sampling, Data Preprocessing, Feature Engineering
- **Seaborn / Matplotlib**: Data Visualisation
- **Scikit-learn**: Machine Learning / Model Evaluation
- **XGBoost** Gradient Boosting Ensemble Modeling
- **Wandb (Weights & Biases)**: For logging machine learning models performance & metrics (for easy visual comparison and complying to data science workflow guidelines)

## 5. Lessons Learnt

## 6. To Start

1. Clone the repo
2. Install packages & libraries via `pip install -r requirements.txt`
3. Run `*.ipynb` files in `notebooks/`
