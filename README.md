# NovaGen Risk Studio

Machine-learning project for NovaGen Research Labs. It compares five classification models on a health-risk dataset and wraps them in an interactive Streamlit app.

Recall is the main metric, because missing a high-risk patient is the costly error.

## Models

| Model                | Accuracy | Recall |
|----------------------|:--------:|:------:|
| Logistic Regression  | 81.4%    | 82.8%  |
| KNN                  | 88.3%    | 88.3%  |
| **Random Forest**    | 93.8%    | 95.9%  |
| Gradient Boosting    | 93.0%    | 94.9%  |
| Voting Classifier    | 91.6%    | 93.1%  |

Random Forest has the highest recall, so it is the recommended model.

## Streamlit app

- **Data explorer:** filters, KPIs and charts of the dataset.
- **Risk assessment:** enter a patient and get a live risk score from any of the five models, with a model consensus chart, a breakdown of what pushes the score up or down, and a what-if curve.
- **Model comparison:** metrics table, adjustable alert threshold, confusion matrix, ROC curves and feature importances.
- **Batch screening:** upload a CSV, score every patient and download the results.

## Project structure

```
novagen_app.py        Streamlit app (trains the models on startup)
novagen_dataset.csv   Dataset (9,549 records, 22 features + Target)
novagen.ipynb         Notebook with the model experiments
requirements.txt      Python dependencies
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run novagen_app.py
```

The app opens at http://localhost:8501. The first load takes about 10 seconds while the models train.

## Tech stack

Python, pandas, NumPy, scikit-learn, Plotly, Streamlit.

## Notes

- Models use an 80/20 stratified train/test split (`random_state=42`).
- This is an educational project. The predictions are not medical advice.