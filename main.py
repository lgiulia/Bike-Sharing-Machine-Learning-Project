# Variabili da ignorare: instant, casual, registered
# Variabili categoriche: season, yr, mnth, hr, weekday, weathersit
# Variabili binarie: holiday, workingday
# Variabili numeriche: temp, atemp, hum, windspeed
# Variabile da trasformare: dteday (usata per estrarre weekday o mnth) ma si può IGNORARE perchè ci sono già nel dataset

import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Caricamento del dataset
df = pd.read_csv('hour.csv') #df: dataframe

print(df.info()) #riepilogo del dataframe
print('\nNull elements: ', df.isnull().values.any()) #verifica l'esistenza di valori mancanti

# Fase 1: EDA e Feature Engeneering
# variabile target y = 'cnt'
y = df['cnt']

#Variabili indipendenti X
#Si ignorano 'instant', 'dteday', 'casual' e 'registered'
X = df.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1) #axis=1: opera sulle colonne

#One-Hot Encoding per le variabili categoriche
#Variabili categoriche da trasformare: 'season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit'
X = pd.get_dummies(X, columns=['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit'], drop_first=True) #converte colonne categoriche in formato numerico

print("\nDataframe dopo One-Hot Encoding:")
print(X.info())

# Fase 2: Visualizzazione dei dati
sns.histplot(x='cnt', data=df, kde=True) #crea un istogramma. kde (Kernel Density Estimate):linea curva
plt.title('Distribution of Bike Rentals')
plt.show()

sns.scatterplot(x='temp', y='cnt', data=df) #crea un grafico a dispersione. Ha una relazione non lineare
plt.title('Bike Rentals vs. Temperature')
plt.show()

#Matrice di correlazione
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Rimuovo 'temp' perché altamente correlata con 'atemp'
X = X.drop(['temp'], axis=1)

# Fase 3: Suddivisione e Scalamento dei Dati
# Divide i dati in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #X_train e y_train: 80%; X_test e y_test: 20%

scaler = StandardScaler() #Standardization (mean=0, std=1)
scaler.fit(X_train) # Calcola i parametri per mean e std sul set di addestramento (X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fase 4: Definizione e Addestramento dei Modelli
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
]

models_names = ['Linear Regression', 'Decision Tree', 'Random Forest']

# Iperparametri per i modelli di regressione
models_parameters = [
    {},  # Linear Regression
    {'max_depth': [5, 10, 15, 20]},  # Decision Tree
    {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},  # Random Forest
]

choosen_hparameters = [] # per salvare i migliori iperparametri
estimators = [] # per salvare i migliori modelli

for model, model_name, hparameters in zip(models, models_names, models_parameters):
    print(f'\n{model_name}')
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='neg_mean_squared_error', cv=5) #estimator: algoritmo che si vuole ottimizzare; param_grid: iperparametri che si vogliono utilizzare;
                                                                                                        #scoring: metrica di valutazione (massimizzare il valore negativo ovvero minimizzare il valore positivo);
                                                                                                        #cv: numero di fold (pieghe) per la cross-validation
    clf.fit(X_train_scaled, y_train) # avvia training e ottimizzazione
    choosen_hparameters.append(clf.best_params_) # combinazione di iperparametri che ha dato il miglior punteggio
    estimators.append((model_name, clf.best_estimator_)) # modello già addestrato con la migliore combinazione di parametri
    best_mse = -clf.best_score_  # conteggio della migliore combinazione di iperparametri. Ho usato ned_mean_squared_error quindi è negativo e si trasforma in positivo
    print(f'Best MSE: {best_mse: .2f}') # errore quadratico medio
    print(f'Best RMSE: {np.sqrt(best_mse):.2f}') # radice dell'errore quadratico medio
    for hparam in hparameters:
        print(f'The best choice for parameter {hparam}: ', clf.best_params_.get(hparam)) # migliori valori trovati per ogni iperparametro

# Stacking Regressor
print("\n--- Stacking Regressor ---")
final_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression()) # Stacking Regressor: combina le previsioni di più modelli di base per creare una previsione finale

# Cross Validation per il modello finale
scores = cross_validate(final_model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error') # final_model: modello che viene addestrato e valutato (Stacking Regressor);
                                                                                                            # X_train_scaled e y_train: dati che vengono utilizzati per la cross-validation (solo training no testing);
                                                                                                            # cv=5: 5 fold (il set di training viene diviso in 5 sotto-insiemi, addestrato su 4 e valutato sul quinto
                                                                                                            # scoring: metrica di valutazione valore negativo di RMSE
rmse_scores = -scores['test_score'] # accede alla chiave 'test_score' nel diz score che contiene i punteggi di RMSE per ogni fold
print(f'The cross-validated RMSE of the Stacking Ensemble meta-model is {np.mean(rmse_scores): .2f}')

# Fase 5: Final Training e Testing
final_model.fit(X_train_scaled, y_train)  # addestra il modello finale (Stacking Regressor) sui dati di X_train_scaled e y_train
y_pred = final_model.predict(X_test_scaled) # il modello fa una previsione sui dati di test. y_pred: viene salvato il risultato delle previsioni

# Valutazione finale
print('\n---- Final Testing Results ----')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)  # percentuale della varianza nel numero di noleggi

print(f'Mean Squared Error (MSE): {mse: .2f}')
print(f'Root Mean Squared Error (RMSE): {rmse: .2f}')
print(f'R-squared (R2) Score: {r2: .2f}')

# Visualizzazione dei risultati
plt.figure(figsize=(12, 10))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Traccia una linea sul grafico che su x e y va dal valore minimo al valore massimo di y_test; r--: riga rossa tratteggiata; lw=2: line width
plt.xlabel('Valori reali')
plt.ylabel('Valori Predetti')
plt.title('Valori Reali vs Valori Predetti')
plt.show()