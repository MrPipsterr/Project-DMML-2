import sys
from functools import partial

import joblib
import pandas as pd
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from designUi.main_window import Ui_Form

class MainForm(QMainWindow):
    model_lgbm = joblib.load('model/model_lgbm.pkl')
    model_lr = joblib.load('model/model_lr.pkl')
    model_cat = joblib.load('model/model_cat.pkl')
    model_bayesian = joblib.load('model/model_bayesian.pkl')
    scaler = joblib.load('model/scaler.pkl')
    current_model = model_cat

    def __init__(self):
        super(MainForm, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # StackWidget
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.homeButton.clicked.connect(self.showHomeWidget)
        self.ui.infoButton.clicked.connect(self.showInfoWidget)

        # Mnunjukkan Value dari Slider Fitur
        self.ui.monsoonIntensity.valueChanged.connect(partial(self.updateLabel, "monsoonIntensityLabel"))
        self.ui.topographyDrainage.valueChanged.connect(partial(self.updateLabel, "topographyDrainageLabel"))
        self.ui.riverManagement.valueChanged.connect(partial(self.updateLabel, "riverManagementLabel"))
        self.ui.deforestation.valueChanged.connect(partial(self.updateLabel, "deforestationLabel"))
        self.ui.urbanization.valueChanged.connect(partial(self.updateLabel, "urbanizationLabel"))
        self.ui.climateChange.valueChanged.connect(partial(self.updateLabel, "climateChangeLabel"))
        self.ui.damsQuality.valueChanged.connect(partial(self.updateLabel, "damsQualityLabel"))
        self.ui.siltation.valueChanged.connect(partial(self.updateLabel, "siltationLabel"))
        self.ui.agriculturalPractices.valueChanged.connect(partial(self.updateLabel, "agriculturalPracticesLabel"))
        self.ui.encroachments.valueChanged.connect(partial(self.updateLabel, "encroachmentsLabel"))
        self.ui.ineffectiveDisasterPreparedness.valueChanged.connect(partial(self.updateLabel, "ineffectiveDisasterPreparednessLabel"))
        self.ui.drainageSystems.valueChanged.connect(partial(self.updateLabel, "drainageSystemsLabel"))
        self.ui.coastalVulnerability.valueChanged.connect(partial(self.updateLabel, "coastalVulnerabilityLabel"))
        self.ui.landslides.valueChanged.connect(partial(self.updateLabel, "landslidesLabel"))
        self.ui.watersheds.valueChanged.connect(partial(self.updateLabel, "watershedsLabel"))
        self.ui.deterioratingInfrastructure.valueChanged.connect(partial(self.updateLabel, "deterioratingInfrastructureLabel"))
        self.ui.populationScore.valueChanged.connect(partial(self.updateLabel, "populationScoreLabel"))
        self.ui.wetlandLoss.valueChanged.connect(partial(self.updateLabel, "wetlandLossLabel"))
        self.ui.inadequatePlanning.valueChanged.connect(partial(self.updateLabel, "inadequatePlanningLabel"))
        self.ui.politicalFactors.valueChanged.connect(partial(self.updateLabel, "politicalFactorsLabel"))

        # ComboBox Connect
        self.ui.chooseAlgorithm.currentIndexChanged.connect(self.on_algorithm_changed)

        # Button Predict
        self.ui.predictButton.clicked.connect(self.on_predict)

    def showHomeWidget(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def showInfoWidget(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def updateLabel(self, label_name, value):
        fiturLabel = getattr(self.ui, label_name)
        fiturLabel.setText(f'{value}')


    BASE_FEATURES = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
        'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
        'Siltation', 'AgriculturalPractices', 'Encroachments',
        'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds',
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
        'InadequatePlanning', 'PoliticalFactors'
    ]

    def on_algorithm_changed(self, index):
        if index == 0:
            self.current_model = self.model_cat
        elif index == 1:
            self.current_model = self.model_lgbm
        elif index == 2:
            self.current_model = self.model_lr
        elif index == 3:
            self.current_model = self.model_bayesian


    def make_prediction(self, features):
        try:
            df = pd.DataFrame([features], columns=self.BASE_FEATURES)
            df = self.add_features(df)
            X = self.scaler.transform(df)
            prediction = self.current_model.predict(X)
            return prediction[0]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error making prediction: {e}")
            return None

    def add_features(self, df):
        df['total'] = df[self.BASE_FEATURES].sum(axis=1)
        df['amplified_sum'] = (df[self.BASE_FEATURES] ** 1.5).sum(axis=1)
        df['fskew'] = df[self.BASE_FEATURES].skew(axis=1)
        df['fkurtosis'] = df[self.BASE_FEATURES].kurtosis(axis=1)
        df['mean'] = df[self.BASE_FEATURES].mean(axis=1)
        df['std'] = df[self.BASE_FEATURES].std(axis=1)
        df['max'] = df[self.BASE_FEATURES].max(axis=1)
        df['min'] = df[self.BASE_FEATURES].min(axis=1)
        df['range'] = df['max'] - df['min']
        df['median'] = df[self.BASE_FEATURES].median(axis=1)
        df['ptp'] = df[self.BASE_FEATURES].values.ptp(axis=1)
        df['q25'] = df[self.BASE_FEATURES].quantile(0.25, axis=1)
        df['q75'] = df[self.BASE_FEATURES].quantile(0.75, axis=1)
        return df

    def on_predict(self):
        try:
            features = [
                self.ui.monsoonIntensity.value(),
                self.ui.topographyDrainage.value(),
                self.ui.riverManagement.value(),
                self.ui.deforestation.value(),
                self.ui.urbanization.value(),
                self.ui.climateChange.value(),
                self.ui.damsQuality.value(),
                self.ui.siltation.value(),
                self.ui.agriculturalPractices.value(),
                self.ui.encroachments.value(),
                self.ui.ineffectiveDisasterPreparedness.value(),
                self.ui.drainageSystems.value(),
                self.ui.coastalVulnerability.value(),
                self.ui.landslides.value(),
                self.ui.watersheds.value(),
                self.ui.deterioratingInfrastructure.value(),
                self.ui.populationScore.value(),
                self.ui.wetlandLoss.value(),
                self.ui.inadequatePlanning.value(),
                self.ui.politicalFactors.value()
            ]
            prediction = self.make_prediction(features)
            if prediction is not None:
                self.ui.resultPredict.setText(f'{prediction:.5f}')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in on_predict: {e}")
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/logo.ico'))
    ex = MainForm()
    ex.setWindowTitle("Flood Prediction")
    ex.show()
    sys.exit(app.exec_())
