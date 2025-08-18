# Variabili da ignorare: instant, casual, registered
# Variabili categoriche: season, yr, mnth, hr, weekday, weathersit
# Variabili binarie: holiday, workingday
# Variabili numeriche: temp, atemp, hum, windspeed
# Variabile da trasformare: dteday (usata per estrarre weekday o mnth) ma di può IGNORARE perchè ci sono già nel dataset

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importazioni per la Regressione
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector

# Modelli di Regressione
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

# Metriche di Valutazione per la Regressione
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Carica il dataset, supponendo che il nome del file sia 'bike_sharing.csv'
df = pd.read_csv('bike+sharing+dataset/hour.csv')

print(df.info())
print(df.describe().T.to_string())
print('\nNull elements: ', df.isnull().values.any())

# Fase 1: EDA e Feature Engineering

# La variabile target (y) è 'cnt'
y = df['cnt']

# Variabili indipendenti (X). Ignoriamo 'instant', 'dteday', 'casual' e 'registered'
# Le variabili categoriche come 'season', 'yr', 'mnth' ecc. saranno trattate con get_dummies
# Le variabili binarie 'holiday' e 'workingday' possono essere usate direttamente
X = df.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'], axis=1)

# Ingegneria delle Caratteristiche: One-Hot Encoding per le variabili categoriche
# Usiamo pd.get_dummies per convertire le colonne categoriche in formato numerico
# Usiamo 'drop_first=True' per evitare la multicollinearità (dummy variable trap)
X = pd.get_dummies(X, columns=['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit'], drop_first=True)

print("\nDataframe dopo One-Hot Encoding:")
print(X.info())
print(X.head())

# Fase 2: Visualizzazione dei dati (Regressione)

# Distribuzione del target
sns.histplot(x='cnt', data=df, kde=True)
plt.title('Distribution of Bike Rentals')
plt.show()

# Esempio di grafico a dispersione per vedere la relazione tra temperatura e noleggi
# Usiamo il dataframe originale per la visualizzazione, è più leggibile
sns.scatterplot(x='temp', y='cnt', data=df)
plt.title('Bike Rentals vs. Temperature')
plt.show()

# Matrice di correlazione per le variabili numeriche originali
# Questa visualizzazione è utile per capire le relazioni iniziali
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Original Numerical Features')
plt.show()

# Fase 3: Suddivisione e Scalamento dei Dati
# Ora usiamo X che include le feature categoriche 'dummy'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fase 4: Definizione e Addestramento dei Modelli
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
]

models_names = ['Linear Regression', 'Decision Tree', 'Random Forest']

# Iperparametri per i modelli di Regressione
models_hparameters = [
    {},  # Linear Regression
    {'max_depth': [5, 10, 15, 20]},
    {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
]

choosen_hparameters = []
estimators = []

for model, model_name, hparameters in zip(models, models_names, models_hparameters):
    print(f'\n{model_name}')
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='neg_mean_squared_error', cv=5)
    clf.fit(X_train_scaled, y_train)
    choosen_hparameters.append(clf.best_params_)
    estimators.append((model_name, clf.best_estimator_))
    best_mse = -clf.best_score_
    print(f'Best MSE: {best_mse:.2f}')
    print(f'Best RMSE: {np.sqrt(best_mse):.2f}')
    for hparam in hparameters:
        print(f'\t The best choice for parameter {hparam}: ', clf.best_params_.get(hparam))

# Stacking Regressor
print("\n---- Stacking Regressor ---- ")
final_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Cross Validation per il modello finale
scores = cross_validate(final_model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -scores['test_score']
print('The cross-validated RMSE of the Stacking Ensemble meta-model is ', np.mean(rmse_scores))

# Selezione delle feature (se necessario)
# Questo passaggio può essere molto lento con un dataset grande e molti features
# Potrebbe essere utile commentarlo o usarlo solo per test
# sfs = SequentialFeatureSelector(final_model, cv=2)
# sfs.fit(X_train_scaled, y_train)
# print('\n ---- Feature Selection ---- ')
# print('Feature selezionate: ', sfs.get_support())
# X_train_final = sfs.transform(X_train_scaled)
# X_test_final = sfs.transform(X_test_scaled)

# Per ora, usiamo semplicemente i dati scalati senza SFS
X_train_final = X_train_scaled
X_test_final = X_test_scaled

# Fase 5: Final Training e Testing
final_model.fit(X_train_final, y_train)
y_pred = final_model.predict(X_test_final)

# Valutazione finale
print('\n---- Final Testing RESULTS -----')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')

# Visualizzazione dei risultati
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valori Reali')
plt.ylabel('Valori Predetti')
plt.title('Valori Reali vs. Predetti')
plt.show()