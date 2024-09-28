# Bankruptcy Prediction in Poland

## Overview

This project aims to predict the likelihood of bankruptcy for companies in Poland using historical financial data. The project utilizes machine learning models to assess a company’s financial health based on various financial ratios and features. The goal is to provide insights that can help companies and stakeholders identify financial risks and take preventive measures.

## Project Highlights
- **Objective**: Build a predictive model that assesses a company’s bankruptcy risk using historical financial data.
- **Data Source**: Dataset collected by Polish economists, containing financial information of companies over several years.
- **Tools Used**: Python, Pandas, scikit-learn, matplotlib, seaborn, and Linux command line.
- **Techniques**: Logistic regression, resampling techniques, decision trees, classification metrics, and data wrangling.

## Workflow
1. **Data Collection**: 
   - Accessed a publicly available dataset from Polish economists studying corporate bankruptcy.
   - Loaded and processed the data using the Linux command line and Python.

2. **Data Preprocessing**:
   - Performed extensive data cleaning using Pandas, addressing missing values and outliers.
   - Handled imbalanced data using oversampling techniques to ensure that the model could accurately predict minority cases of bankruptcy.
   - Conducted exploratory data analysis (EDA) to identify key financial ratios and features that influence bankruptcy risk.

3. **Modeling**:
   - Built multiple classification models, including **logistic regression** and **decision tree classifiers**, to predict bankruptcy risk.
   - Fine-tuned model hyperparameters using grid search and cross-validation to optimize performance.
   - Evaluated models using classification metrics such as accuracy, precision, recall, and F1-score to determine the best-performing model.

4. **Prediction**:
   - Deployed the best-performing model to predict the likelihood of bankruptcy for new companies.
   - Visualized prediction results to highlight the most critical factors influencing bankruptcy.

## Results
- The **logistic regression model** performed well in predicting bankruptcy, with high precision and recall for the positive (bankruptcy) class.
- The model was able to capture significant financial trends and patterns that indicate a company’s potential risk of bankruptcy.
- The use of resampling techniques helped in effectively handling the imbalanced nature of the dataset.

## Key Learnings
- **Financial Analysis**: Gained deeper insights into how financial ratios and key indicators can affect a company's bankruptcy risk.
- **Data Imbalance**: Successfully handled imbalanced datasets using resampling techniques to ensure that the minority class (bankruptcy cases) was adequately represented in the model.
- **Machine Learning**: Improved skills in building and optimizing classification models for real-world financial problems, with a focus on logistic regression and decision trees.
  
## Future Improvements
- **Incorporate Time-Series Data**: Add historical data trends and time-series analysis to improve the model’s ability to predict long-term bankruptcy risks.
- **Advanced Ensemble Models**: Experiment with advanced ensemble models such as Random Forest, Gradient Boosting, or XGBoost to improve prediction accuracy.
- **Interactive Dashboard**: Develop a user-friendly dashboard that allows businesses to input their financial data and receive real-time bankruptcy risk predictions.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/nisha2k21/bankruptcy-prediction-poland.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd bankruptcy-prediction-poland
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the model to make predictions:
   ```bash
   python src/bankruptcy_prediction.py
   ```

## Project Structure
```
├── data
│   └── bankruptcy_data.csv          # Raw data file
├── notebooks
│   └── EDA_and_Modeling.ipynb       # Jupyter notebook for analysis and model building
├── src
│   └── bankruptcy_prediction.py     # Python script for running the model
├── requirements.txt                 # List of required dependencies
└── README.md                        # Project documentation
```

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Machine Learning**: Logistic regression, Decision tree classifiers, Resampling techniques
- **Model Evaluation**: Precision, Recall, F1-score, Accuracy
- **Platform**: Linux command line for data handling

## Contact
For any questions, feedback, or collaboration opportunities, feel free to reach out:
- **Email**: nisha2k21@gmail.com
- **LinkedIn**: [Nisha Kumari](https://www.linkedin.com/in/nisha-kumari-041300225/)
