import pandas as pd
import numpy as np

# Function that returns a dictionary containing 
# the number of occurrences of the dataset for each columns
def count_unique_values(df):
    unique_values = {}
    for col in df.drop(df.columns[-1], axis=1).columns:
        unique_values[col] = df[col].value_counts().to_dict()
    return unique_values

def Likelihood(train_set, v, df):
    prob_yes = {}
    prob_no = {}
    for col in train_set:
        prob_yes[col] = train_set[df.columns[-1]].eq(2).groupby(train_set[col]).sum().to_dict()
        prob_no[col] = train_set[df.columns[-1]].eq(1).groupby(train_set[col]).sum().to_dict()
    prob_yes.popitem()
    prob_no.popitem()

    def LaplaceSmoothing(prob_yes, prob_no, v):
        for col in prob_yes.keys():
            a=0.85
            x=0
            for k in range(1, len(count_unique_values(df)[col])+1):
                if k not in prob_yes[col]:
                    prob_yes[col][k] = 0
                if k not in prob_no[col]:
                    prob_no[col][k] = 0
                prob_yes[col][k] =(prob_yes[col][k]+a)/(sum(prob_yes[col].values())+a*v[x])
                prob_no[col][k]=(prob_no[col][k]+a)/(sum(prob_no[col].values())+a*v[x])
            x+=1
        return(prob_yes, prob_no)
    
    prob_yes, prob_no = LaplaceSmoothing(prob_yes, prob_no, v)
    return(prob_yes, prob_no)

def PriorProbability(df):
    prob_yes_no = df[df.columns[-1]].value_counts(normalize=True).to_dict()
    for k in range(1, 3):
            if k not in prob_yes_no:
                prob_yes_no[k] = 0
    return prob_yes_no

def NaiveBayesClassifier(train_set, test_set, df):

    def FinalProbability(test_set, prob_yes, prob_no):
        final_yes = {}
        final_no = {}
        for i in range(test_set.shape[0]):
            mul_yes = 1
            mul_no = 1
            for col in test_set.columns:
                mul_yes = mul_yes * prob_yes[col][test_set.iloc[i][col]]
                mul_no = mul_no * prob_no[col][test_set.iloc[i][col]]
            final_yes[i] = mul_yes * PriorProbability(df)[2]
            final_no[i] = mul_no * PriorProbability(df)[1]
        return(final_yes, final_no)
    
    def Predictions(final_yes, final_no):
        prediction = list()
        for i in range(len(final_yes)):
            if (final_yes[i] >= final_no[i]):
                prediction.append(2)
            else:
                prediction.append(1)
        return prediction
    
    test_play = list(test_set[df.columns[-1]])
    test_set.pop(df.columns[-1])
    prob_yes, prob_no = Likelihood(train_set, v, df)
    final_yes, final_no = FinalProbability(test_set, prob_yes, prob_no)
    prediction = Predictions(final_yes, final_no)
    
    sum = 0
    for i in range(len(prediction)):
        if (prediction[i] == test_play[i]):
            sum+=1

    errore = int((1 - sum/len(test_play))*100)
    return(test_play, prediction, errore)

# Read the DataSet file
df=pd.read_csv('Lab1_data.csv', delim_whitespace=True) 

# Convert Categorical data into Numerical data
for col in df.columns:
    class_map ={label: idx for idx, label in enumerate(np.unique(df[col]))}
    df[col] = df[col].map(class_map)

# Split the DataSet in training set (75%) and test set (25%)
df1 = df.sample(frac=1) + 1
head = int((df1.shape[0] * 75)/100)
tail = df1.shape[0] - head
train_X = df1.head(head)
test_X = df1.tail(tail)

# Vector that counts the number of columns' unique values of the dataset
v = list(df1.nunique())
v.pop(4)

# Call to the NaiveBayesClassifier function and results
test_play, prediction, errore = NaiveBayesClassifier(train_X, test_X, df1)
print('')
print('Prediction result:', prediction)
print('Actual result:', test_play, '\n')
print('Error Rate:', errore,'%', '\n')

