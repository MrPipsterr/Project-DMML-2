import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


def add_features(df):
    BASE_FEATURES = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
                     'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
                     'Siltation', 'AgriculturalPractices', 'Encroachments',
                     'IneffectiveDisasterPreparedness', 'DrainageSystems',
                     'CoastalVulnerability', 'Landslides', 'Watersheds',
                     'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                     'InadequatePlanning', 'PoliticalFactors']
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

train = pd.read_csv('../Data/train.csv')

train = add_features(train)

NON_FEATURES = ['id', 'FloodProbability', 'fold']
FEATURES = [col for col in train.columns if col not in NON_FEATURES]

X_train = train[FEATURES]
y_train = train['FloodProbability']

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

joblib.dump(scaler, '../model/scaler.pkl')

model_lgbm = LGBMRegressor(objective='regression', random_state=0)
model_lgbm.fit(X_train, y_train)

joblib.dump(model_lgbm, '../model/model_lgbm.pkl')