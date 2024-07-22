# Classificatore Naive Bayes

## Descrizione dell'Assegnamento

L'assegnamento consiste in tre compiti principali:

- **Compito 1:** Pre-elaborazione dei dati
- **Compito 2:** Costruire un classificatore Naive Bayes
- **Compito 3:** Migliorare il classificatore con la lisciatura di Laplace (additiva)

## Compito 1: Pre-elaborazione dei dati

Il nostro dataset è inizialmente memorizzato come file `.csv`. Per importarlo in Python, utilizziamo la funzione `read_csv()` della libreria Pandas per memorizzare i dati in un dataframe. È consigliabile convertire i valori degli attributi in valori interi positivi per facilitarne la manipolazione.

Il set di dati utilizzato è un set di dati meteorologici composto da:

- **14 osservazioni**
- **4 attributi:** outlook, temperature, humidity, windy
- **2 classi:** play YES, play NO

Nel dataset, le prime quattro colonne rappresentano gli attributi, mentre l'ultima colonna è il target dell'osservazione. In particolare:

- `outlook` può assumere tre valori: overcast, rainy o sunny
- `temperature` può assumere tre valori: hot, mild o cool
- `humidity` può assumere due valori: high o normal
- `windy` può assumere due valori: true o false

### Codice di Pre-elaborazione

```python
import pandas as pd

# Carica il dataset
df = pd.read_csv('weather.csv')

# Converti i valori degli attributi in valori interi positivi
# (Codifica degli attributi e delle classi se necessario)
```
## Compito 2: Costruzione di un Classificatore Naive Bayes

Il compito consiste nella costruzione di un classificatore Naive Bayes, suddiviso in due fasi principali: **fase di addestramento** e **fase di classificazione**. Per calcolare la classificazione, abbiamo implementato una funzione chiamata `NaiveBayesClassifier()`, che utilizza altre funzioni ausiliarie per eseguire i calcoli necessari.

### 1. Fase di Addestramento

Durante la fase di addestramento, calcoliamo le probabilità necessarie per classificare le osservazioni. Le funzioni principali coinvolte sono:

#### 2.1 Calcolo della Likelihood

La funzione `Likelihood()` calcola le occorrenze dei valori "Yes" (valore = 2) e "No" (valore = 1) nella colonna target. Restituisce due liste contenenti le occorrenze per ciascuno dei quattro attributi (outlook, temperature, humidity, windy) in base al valore della colonna target.

- **Funzione:** `Likelihood()`
- **Descrizione:** Calcola le occorrenze per ogni attributo, suddiviso per il valore della colonna target.
- **Funzione ausiliaria:** Chiama `LaplaceSmoothing()` per applicare la correzione di Laplace (discussa nel Compito 3).

**Codice per `Likelihood()`:**
```python
def Likelihood(train_set, v, df):
    # Calcola le occorrenze per ogni attributo
    # ...
    return occurrences_yes, occurrences_no
```
### 2.2 Calcolo della Probabilità Finale

La funzione `FinalProbability()` classifica ogni osservazione del set di test utilizzando la regola di massimizzazione della funzione discriminante \( g_i(x) \). Per calcolare queste probabilità, la funzione chiama anche `PriorProbability()` per determinare le probabilità a priori di "Yes" e "No".

- **Funzione:** `FinalProbability()`
- **Descrizione:** Calcola le probabilità finali di "Yes" e "No" per ogni osservazione del set di test.
- **Funzione ausiliaria:** Chiama `PriorProbability()` per ottenere le probabilità a priori.

**Codice per `FinalProbability()`:**
```python
def FinalProbability(test_set, occurrences_yes, occurrences_no):
    # Calcola le probabilità finali di Yes e No
    # ...
    return final_yes, final_no
```
### 2.3 Calcolo delle Predizioni

La funzione `Predictions()` confronta le probabilità finali di "Yes" e "No" per determinare la previsione. Se la probabilità di "Yes" è maggiore, la previsione sarà "Yes" (valore: 2); altrimenti, sarà "No" (valore: 1).

- **Funzione:** `Predictions()`
- **Descrizione:** Confronta le probabilità finali di "Yes" e "No" per fare la previsione per ogni osservazione.

**Codice per `Predictions()`:**
```python
def Predictions(final_yes, final_no):
    # Confronta le probabilità per fare la previsione
    # ...
    return prediction
```
## Compito 3: Migliorare il Classificatore con la Lisciatura di Laplace (Additiva)

Per migliorare le prestazioni del classificatore Naive Bayes, utilizziamo la lisciatura di Laplace (additiva). Questa tecnica inserisce un fattore di lisciatura all'interno dell'equazione di probabilità per gestire le probabilità di eventi rari e migliorare la robustezza del classificatore. Il valore di lisciatura, denotato come \( a \), è specificato nella funzione `LaplaceSmoothing()`. Per la nostra implementazione, abbiamo scelto un valore di \( a = 0.85 \), per ridurre l'affidamento del classificatore sui conteggi a priori rispetto ai dati osservati.


