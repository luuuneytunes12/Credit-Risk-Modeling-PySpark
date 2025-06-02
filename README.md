## 1. Project Description
<br>
<img src="./images/Credit-Risk-Modelling-1.webp" width='100%'>

In this project, I will be exploring the world of credit risk modeling in banks. To simulate this, I will be using a dataset provided by Lending Club, a peer-to-peer lending company. Though slightly different from the business of traditional banks loan consumers, this project would be a good stepping stone for me to learn the fundamentals of credit risk modeling. 

Specifically, i will be building 3 ML models to predict PD (Probability of Default), EAD (Exposure at Default) via tree / regression models and LGD (Loss Given Default) via survival analysis. I will then be deriving an interpretable credit scorecard and Expected Loss for each loan, to assist Lending Club in improving its credit policy. 

This project is inspired by https://github.com/allmeidaapedro/Lending-Club-Credit-Scoring, and I will be incorporating additional ideas in different phases of the project, such as survival analysis and PCA, for learning purposes. 

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

- Lending Club wants to **expand its customer base** by managing default risks effectively. It sees the need to build **highly accurate credit risk models**, and diversify its portfolio across low to high-risk borrowers to maximise its profitability

- **Task:** As a junior data scientist, I will need to **help Lending Club identify the important factors to include in its credit scorecard, by constructing accurate credit risk models**

- Credit scorecards will be used to allocate risk levels to borrowers. Should credit scores fall below a certain threshold, their loans shall be denied by the Lending Club platform even if lenders are willing to lend their capital, with strict accordance to Basel III regulations. 


##### 2.3 Project Objectives
  1. Build a highly accurate **Probability of Default (PD) Model**, developing a **credit scorecard** for ease of interpretation. Decision whether to grant a loan will be based on credit scores 

  2. Build **Exposure at Default (EAD) Model** and **Loss Given Default (LGD) Model** via regression / tree / survival analysis models, estimating **Expected Loss (EL)** in loans. This ensures Lending Club has sufficient capital to adhere to strict Basel III regulations at any given point, and also to prevent itself from defaulting. 

  3. Establish a well justified **credit policy**, e.g. 'Only approve if Expected Loss < $20 000'

  4. Apply comprehensive **model evaluation & stress testing techniques** to comply to strict Basel III financial regulations 


## 3. Project Structure 
- **images/**: Contains images needed for storytelling and visualisations 
- **notebooks/**: Contains Jupyter notebooks for data preprocessing, model building and model evaluation 
- **artifacts/**: Contains ready-made machine learning models 


## 4. Tech Stack 
- **PySpark** (Big Data processing)
- **PySurvival** (Survival analysis)
- **Imbalanced-learn** (SMOTE)
- **Scikit-learn** (PCA/LDA)
- **XGBoost** (Gradient boosting) - 待定
- Numpy... 


