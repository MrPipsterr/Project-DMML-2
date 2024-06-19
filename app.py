import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model/model_lgbm.pkl')
scaler = joblib.load('model/scaler.pkl')

BASE_FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

def make_prediction(features):
    df = pd.DataFrame([features], columns=BASE_FEATURES)
    df = add_features(df)
    X = scaler.transform(df)
    prediction = model.predict(X)
    return prediction[0]

def add_features(df):
    df['total'] = df[BASE_FEATURES].sum(axis=1)
    df['amplified_sum'] = (df[BASE_FEATURES] ** 1.5).sum(axis=1)
    df['fskew'] = df[BASE_FEATURES].skew(axis=1)
    df['fkurtosis'] = df[BASE_FEATURES].kurtosis(axis=1)
    df['mean'] = df[BASE_FEATURES].mean(axis=1)
    df['std'] = df[BASE_FEATURES].std(axis=1)
    df['max'] = df[BASE_FEATURES].max(axis=1)
    df['min'] = df[BASE_FEATURES].min(axis=1)
    df['range'] = df['max'] - df['min']
    df['median'] = df[BASE_FEATURES].median(axis=1)
    df['ptp'] = df[BASE_FEATURES].values.ptp(axis=1)
    df['q25'] = df[BASE_FEATURES].quantile(0.25, axis=1)
    df['q75'] = df[BASE_FEATURES].quantile(0.75, axis=1)
    return df

def on_predict():
    features = [slider.get() for slider in sliders]
    prediction = make_prediction(features)
    result_label.config(text=f'Predicted FloodProbability: {prediction:.5f}')

def increase(slider):
    value = slider.get()
    if value < 20:
        slider.set(value + 1)

def decrease(slider):
    value = slider.get()
    if value > 0:
        slider.set(value - 1)

def main():
    root = tk.Tk()
    root.title("Flood Prediction")

    root.state('zoomed')

    features = BASE_FEATURES

    global sliders
    sliders = []

    canvas = tk.Canvas(root)
    scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set)

    for i, feature in enumerate(features):
        if i % 2 == 0:
            row_frame = tk.Frame(frame)
            row_frame.pack(fill="x", pady=5)

        col_frame = tk.Frame(row_frame)
        col_frame.pack(side=tk.LEFT, padx=10)

        label = tk.Label(col_frame, text=feature, width=20)
        label.grid(row=0, column=0, padx=5, pady=5)

        slider = tk.Scale(col_frame, from_=0, to=20, orient=tk.HORIZONTAL)
        slider.set(10)

        minus_button = tk.Button(col_frame, text="-", command=lambda s=slider: decrease(s), bg='red', fg='white')
        minus_button.grid(row=0, column=1, padx=5, pady=5)

        slider.grid(row=0, column=2, padx=5, pady=5)
        sliders.append(slider)

        plus_button = tk.Button(col_frame, text="+", command=lambda s=slider: increase(s), bg='green', fg='white')
        plus_button.grid(row=0, column=3, padx=5, pady=5)

    right_frame = tk.Frame(root)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    predict_button = tk.Button(right_frame, text="Predict", command=on_predict, bg='blue', fg='white')
    predict_button.pack(fill='x')

    global result_label
    result_label = tk.Label(right_frame, text="Predicted FloodProbability: ", font=("Helvetica", 14))
    result_label.pack(fill='x', pady=10)

    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scroll_y.pack(side=tk.RIGHT, fill="y")

    root.mainloop()

if __name__ == "__main__":
    main()
