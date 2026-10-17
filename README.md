# NovaGen Risk Studio

Machine-learning project for NovaGen Research Labs. It compares three classification models on a health-risk dataset and wraps them in an interactive Streamlit app.

Recall is the main metric, because missing a high-risk patient is the costly error.

## Models

| Model                | Recall | Accuracy |
|----------------------|:------:|:--------:|
| Logistic Regression  | 82.8%  | 81.4%    |
| **Random Forest**    | 95.9%  | 93.8%    |
| Gradient Boosting    | 95.0%  | 93.0%    |

- **Logistic Regression:** simple linear baseline with L2 regularization (uses scaled features).
- **Random Forest:** bagging, many decision trees trained on random samples and averaged.
- **Gradient Boosting:** boosting, trees built one after another, each correcting the previous errors.

Random Forest has the highest recall and is the recommended model. It is only about one percentage point ahead of Gradient Boosting on a single train/test split.

## Streamlit app

- **Data explorer:** filters, KPIs and charts of the dataset.
- **Risk assessment:** enter a patient and get a live risk score from any of the three models, with a model consensus chart, a breakdown of what pushes the score up or down, and a what-if curve.
- **Model comparison:** metrics table, adjustable alert threshold, confusion matrix, ROC curves and feature importances.
- **Batch screening:** upload a CSV, score every patient and download the results.

## Project structure

```
novagen_app.py        Streamlit app (trains the models on startup)
novagen_dataset.csv   Dataset (9,549 records, 22 features + Target)
novagen_1.ipynb       Notebook with the model experiments
requirements.txt      Python dependencies
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run novagen_app.py
```

The app opens at http://localhost:8501. The first load takes a few seconds while the models train.

## Tech stack

Python, pandas, NumPy, scikit-learn, Plotly, Streamlit.

## Notes

- Models use an 80/20 stratified train/test split (`random_state=42`).
- The dataset appears to be synthetic, so the project focuses on comparing models rather than drawing medical conclusions.
- This is an educational project. The predictions are not medical advice.