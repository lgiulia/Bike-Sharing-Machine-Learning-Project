# Variabili da ignorare: instant, casual, registered
# Variabili categoriche: season, yr, mnth, hr, weekday, weathersit
# Variabili binarie: holiday, workingday
# Variabili numeriche: temp, atemp, hum, windspeed
# Variabile da trasformare: dteday (usata per estrarre weekday o mnth) ma si può IGNORARE perchè ci sono già nel dataset

import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Caricamento del dataset
df = pd.read_csv('hour.csv') #df: dataframe

print(df.info()) #riepilogo del dataframe
print("\n")
print(df.head()) #stampa le prime 5 righe
print("\n")
print(df.describe().T.to_string()) #riassunto statistico di tutte le variabili numeriche
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
print("\n")
print(X.head()) #stampa le prime 5 righe del dataframe

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

# Fase 3: Suddivisione e Scalamento dei Dati
# Divide i dati i training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #X_train e y_train: 80%; X_test e y_test: 20%

scaler = StandardScaler() #Standardization (mean=0, std=1)
scaler.fit(X_train) # Calcola i parametri per mean e std sul set di addestramento (X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fase 4: Definizione e Addestramento dei Modelli
