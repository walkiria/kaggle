import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import classifiers


import re
import sys



import statsmodels.api as sm
from statsmodels.formula.api import ols


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from skfeature.function.similarity_based import fisher_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier  # use RandomForestRegressor for regression problem

from sklearn.metrics import confusion_matrix

df = pd.read_csv("/home/walkiria/workspace/kaggle/titanic/train.csv", index_col=0, sep=",")
df_teste = pd.read_csv("/home/walkiria/workspace/kaggle/titanic/test.csv", index_col=0, sep=",")


# train
individuals, features = np.shape(df)
X = df.iloc[0:individuals, 1:features]
y = df.iloc[0:individuals, 0]
y_case_control = pd.crosstab(y,columns="counts")

# test

individuals_test, features_test = np.shape(df_teste)
X_test = df_teste.iloc[0:individuals, 0:features]

# calcular as caracteristicas que são iguais tanto para caso quanto para controle usando chi-squared



char_1 =    pd.Categorical(X.iloc[0:individuals, 0].astype(str), categories=['1', '2', '3'])
classe = pd.Categorical(y.astype(str), categories=['0', '1'])
count_freq = pd.crosstab( char_1,  classe)

char_2 =    X.iloc[0:individuals, 2]
count_freq = pd.crosstab( char_2,  classe)


char_3 = X.iloc[0:individuals, 9]
count_freq = pd.crosstab( char_3,  classe)


# PCLASSE

expected_case_1 = (sum(count_freq.iloc[0]) * sum(count_freq.iloc[0:3, 1]) ) / np.shape(y)
expected_control_1 = sum(count_freq.iloc[0]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)

expected_case_2 = sum(count_freq.iloc[1]) * sum(count_freq.iloc[0:3, 1]) / np.shape(y)
expected_control_2 = sum(count_freq.iloc[1]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)

expected_case_3 = sum(count_freq.iloc[2]) * sum(count_freq.iloc[0:3, 1]) / np.shape(y)
expected_control_3 = sum(count_freq.iloc[2]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)



chi_squared_stat_case_1 = ((count_freq.iloc[0, 1]-expected_case_1)**2)/expected_case_1
chi_squared_stat_control_1 = (((count_freq.iloc[0, 0]-expected_control_1)**2)/expected_control_1)

chi_squared_stat_case_2 = (((count_freq.iloc[1, 1]-expected_case_2)**2)/expected_case_2)
chi_squared_stat_control_2 = (((count_freq.iloc[1, 0]-expected_control_2)**2)/expected_control_2)

chi_squared_stat_case_3 = (((count_freq.iloc[2, 1]-expected_case_3)**2)/expected_case_3)
chi_squared_stat_control_3 = (((count_freq.iloc[2, 0]-expected_control_3)**2)/expected_control_3)


chi_squared_total = chi_squared_stat_case_1 + chi_squared_stat_control_1 + chi_squared_stat_case_2 +  chi_squared_stat_control_2 + chi_squared_stat_case_3 +chi_squared_stat_control_3

graus_liberdade = (3- 1) * (2- 1)

# checar tabela de chisquared ==array([ 102.88898876])
#  como o calculado é maior que o tabelado os desvios sao significativos


# SEXO

expected_case_1 = (sum(count_freq.iloc[0]) * sum(count_freq.iloc[0:2, 1]) ) / np.shape(y)
expected_control_1 = sum(count_freq.iloc[0]) *sum(count_freq.iloc[0:2, 0]) /  np.shape(y)

expected_case_2 = sum(count_freq.iloc[1]) * sum(count_freq.iloc[0:2, 1]) / np.shape(y)
expected_control_2 = sum(count_freq.iloc[1]) *sum(count_freq.iloc[0:2, 0]) /  np.shape(y)


chi_squared_stat_case_1 = ((count_freq.iloc[0, 1]-expected_case_1)**2)/expected_case_1
chi_squared_stat_control_1 = (((count_freq.iloc[0, 0]-expected_control_1)**2)/expected_control_1)

chi_squared_stat_case_2 = (((count_freq.iloc[1, 1]-expected_case_2)**2)/expected_case_2)
chi_squared_stat_control_2 = (((count_freq.iloc[1, 0]-expected_control_2)**2)/expected_control_2)

chi_squared_total_sex = chi_squared_stat_case_1 + chi_squared_stat_control_1 + chi_squared_stat_case_2 +  chi_squared_stat_control_2

graus_liberdade = (2-1) * (2-1)

# os desvios sao significativos   -- 263.05057407


# EMBARQUE EM S, C, Q


expected_case_1 = (sum(count_freq.iloc[0]) * sum(count_freq.iloc[0:3, 1]) ) / np.shape(y)
expected_control_1 = sum(count_freq.iloc[0]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)

expected_case_2 = sum(count_freq.iloc[1]) * sum(count_freq.iloc[0:3, 1]) / np.shape(y)
expected_control_2 = sum(count_freq.iloc[1]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)

expected_case_3 = sum(count_freq.iloc[2]) * sum(count_freq.iloc[0:3, 1]) / np.shape(y)
expected_control_3 = sum(count_freq.iloc[2]) *sum(count_freq.iloc[0:3, 0]) /  np.shape(y)



chi_squared_stat_case_1 = ((count_freq.iloc[0, 1]-expected_case_1)**2)/expected_case_1
chi_squared_stat_control_1 = (((count_freq.iloc[0, 0]-expected_control_1)**2)/expected_control_1)

chi_squared_stat_case_2 = (((count_freq.iloc[1, 1]-expected_case_2)**2)/expected_case_2)
chi_squared_stat_control_2 = (((count_freq.iloc[1, 0]-expected_control_2)**2)/expected_control_2)

chi_squared_stat_case_3 = (((count_freq.iloc[2, 1]-expected_case_3)**2)/expected_case_3)
chi_squared_stat_control_3 = (((count_freq.iloc[2, 0]-expected_control_3)**2)/expected_control_3)


chi_squared_total_embarque = chi_squared_stat_case_1 + chi_squared_stat_control_1 + chi_squared_stat_case_2 +  chi_squared_stat_control_2 + chi_squared_stat_case_3 +chi_squared_stat_control_3

graus_liberdade = (3- 1) * (2- 1)


# os desvios sao significativos   -- 26.55323232



### caracteristicas relevantes: sexo, classe e embarque


# checar as caracteristicas numericas -- quantitativas

# checar se é casado/solteiro

#

# testar as variaveis categoricas




######## inicio treino ##########

X= df.iloc[0:individuals, 1:features]

## search by families
X['family'] = np.random.randn(individuals)

split_family = []
for line in X.iloc[0:individuals, 1]:
   split_family.append(re.split(",", line))


i = 0
j= 1

for i in range(0, individuals):
    for j in range(1, individuals):
     if(split_family[i][0] == split_family[j][0]):
         X.iloc[i, 10] = split_family[i][0]
         print(i)
     j = j+1
i = i + 1


for i in range(0, individuals):
    pos = np.where(X.iloc[0:individuals, 10] == X.iloc[i, 10])[0]
    X.iloc[pos, 10] = i



# search by ticket
split_ticket = []
for line in X.iloc[0:individuals, 6]:
   split_ticket.append(re.split("[0-9]*", line))


list = []
i = 0
for i in range(0, individuals):
    list.append(split_ticket[i][0])
    list[i] = list[i].strip()
    list[i] = list[i].replace(".", "")
    list[i] = list[i].replace("/", "")
    i= i+1


ticket = pd.Categorical(list)
ticket_class = pd.crosstab(ticket,  classe)


for i in range(0, individuals):
    if(list[i] == "A"):
        X.iloc[i, 6]= 0
    elif(list[i] == "AS"):
        X.iloc[i, 6]= 1
    elif(list[i] == "C"):
        X.iloc[i, 6]= 2
    elif(list[i] == "CA"):
        X.iloc[i, 6]= 3
    elif(list[i] == "CASOTON"):
        X.iloc[i, 6]= 4
    elif(list[i] == "FC"):
        X.iloc[i, 6]= 5
    elif(list[i] == "FCC"):
        X.iloc[i, 6]= 6
    elif(list[i] == "Fa"):
        X.iloc[i, 6]= 7
    elif(list[i] == "LINE"):
        X.iloc[i, 6]= 8
    elif(list[i] == "PC"):
        X.iloc[i, 6]= 9
    elif(list[i] == "PP"):
        X.iloc[i, 6]= 10
    elif(list[i] == "PPP"):
        X.iloc[i, 6]= 11
    elif(list[i] == "SC"):
        X.iloc[i, 6]= 12
    elif(list[i] == "SCA"):
        X.iloc[i, 6]= 13
    elif(list[i] == "SCAH"):
        X.iloc[i, 6]= 14
    elif(list[i] == "SCAH Basle"):
        X.iloc[i, 6]= 15
    elif(list[i] == "SCOW"):
        X.iloc[i, 6]= 16
    elif(list[i] == "SCPARIS"):
        X.iloc[i, 6]= 17
    elif(list[i] == "SCParis"):
        X.iloc[i, 6]= 17
    elif(list[i] == "SOC"):
        X.iloc[i, 6]= 19
    elif(list[i] == "SOP"):
        X.iloc[i, 6]= 20
    elif(list[i] == "SOPP"):
        X.iloc[i, 6]= 21
    elif(list[i] == "SOTONO"):
        X.iloc[i, 6]= 22
    elif(list[i] == "SOTONOQ"):
        X.iloc[i, 6]= 23
    elif(list[i] == "SP"):
        X.iloc[i, 6]= 24
    elif(list[i] == "STONO"):
        X.iloc[i, 6]= 25
    elif(list[i] == "SWPP"):
        X.iloc[i, 6]= 26
    elif(list[i] == "WC"):
        X.iloc[i, 6]= 27
    elif(list[i] == "WEP"):
        X.iloc[i, 6]= 28
    else:
        X.iloc[i, 6] = -1


X.iloc[0:individuals, 4 ] =  X.iloc[0:individuals, 4 ] +X.iloc[0:individuals, 5 ]

# treatment pronouns

i =0
for line in X.iloc[0:individuals, 1]:
    if re.search("Mr.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 0
    if re.search("Miss.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 1
    if re.search("Master", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 2
    if re.search("Don.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 3
    if re.search("Rev.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 4
    if re.search("Dr.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 5
    if re.search("Mme.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 6
    if re.search("Ms.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 7
    if re.search("Major.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 8
    if re.search("Mlle.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 9
    if re.search("Col.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 10
    if re.search("Capt.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 11
    if re.search("Countess.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 12
    if re.search("Jonkheer.", line):
        print(X.iloc[i, 1])
        X.iloc[i, 1] = 13

    i = i + 1




X.iloc[0:individuals, 2] = X.iloc[0:individuals, 2].replace("male", 0)
X.iloc[0:individuals, 2] = X.iloc[0:individuals, 2].replace("female", 1)


X.iloc[0:individuals, 9] = X.iloc[0:individuals, 9].replace("S", 0)
X.iloc[0:individuals, 9] = X.iloc[0:individuals, 9].replace("Q", 1)
X.iloc[0:individuals, 9] = X.iloc[0:individuals, 9].replace("C", 2)



#Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked


X_1 = X.iloc[0:individuals, [0,  2,  6, 7, 10] ]


#for colunm in X_1:
#    X_1 = X_1.replace("NaN", X_1.mean())


X_1 = X_1.interpolate()

X_1 = X_1.astype(int)


######## fim treino ##########


### inicio teste
X_test = df_teste.iloc[0:individuals_test, 0:features_test]


## search by families
X_test['family'] = np.random.randn(individuals_test)

split_family = []
for line in X_test.iloc[0:individuals_test, 1]:
   split_family.append(re.split(",", line))


i = 0
j= 1

for i in range(0, individuals_test):
    for j in range(1, individuals_test):
     if(split_family[i][0] == split_family[j][0]):
         X_test.iloc[i, 10] = split_family[i][0]
         print(i)
     j = j+1
i = i + 1


for i in range(0, individuals_test):
    pos = np.where(X_test.iloc[0:individuals_test, 10] == X_test.iloc[i, 10])[0]
    X_test.iloc[pos, 10] = i


# search by ticket
split_ticket = []
for line in X_test.iloc[0:individuals_test, 6]:
   split_ticket.append(re.split("[0-9]*", line))


list = []
i = 0
for i in range(0, individuals_test):
    list.append(split_ticket[i][0])
    list[i] = list[i].strip()
    list[i] = list[i].replace(".", "")
    list[i] = list[i].replace("/", "")
    i= i+1


ticket = pd.Categorical(list)


for i in range(0, individuals_test):
    if(list[i] == "A"):
        X_test.iloc[i, 6]= 0
    elif(list[i] == "AS"):
        X_test.iloc[i, 6]= 1
    elif(list[i] == "C"):
        X_test.iloc[i, 6]= 2
    elif(list[i] == "CA"):
        X_test.iloc[i, 6]= 3
    elif(list[i] == "CASOTON"):
        X_test.iloc[i, 6]= 4
    elif(list[i] == "FC"):
        X_test.iloc[i, 6]= 5
    elif(list[i] == "FCC"):
        X_test.iloc[i, 6]= 6
    elif(list[i] == "Fa"):
        X_test.iloc[i, 6]= 7
    elif(list[i] == "LINE"):
        X_test.iloc[i, 6]= 8
    elif(list[i] == "PC"):
        X_test.iloc[i, 6]= 9
    elif(list[i] == "PP"):
        X_test.iloc[i, 6]= 10
    elif(list[i] == "PPP"):
        X_test.iloc[i, 6]= 11
    elif(list[i] == "SC"):
        X_test.iloc[i, 6]= 12
    elif(list[i] == "SCA"):
        X_test.iloc[i, 6]= 13
    elif(list[i] == "SCAH"):
        X_test.iloc[i, 6]= 14
    elif(list[i] == "SCAH Basle"):
        X_test.iloc[i, 6]= 15
    elif(list[i] == "SCOW"):
        X_test.iloc[i, 6]= 16
    elif(list[i] == "SCPARIS"):
        X_test.iloc[i, 6]= 17
    elif(list[i] == "SCParis"):
        X_test.iloc[i, 6]= 17
    elif(list[i] == "SOC"):
        X_test.iloc[i, 6]= 19
    elif(list[i] == "SOP"):
        X_test.iloc[i, 6]= 20
    elif(list[i] == "SOPP"):
        X_test.iloc[i, 6]= 21
    elif(list[i] == "SOTONO"):
        X_test.iloc[i, 6]= 22
    elif(list[i] == "SOTONOQ"):
        X_test.iloc[i, 6]= 23
    elif(list[i] == "SP"):
        X_test.iloc[i, 6]= 24
    elif(list[i] == "STONO"):
        X_test.iloc[i, 6]= 25
    elif(list[i] == "SWPP"):
        X_test.iloc[i, 6]= 26
    elif(list[i] == "WC"):
        X_test.iloc[i, 6]= 27
    elif(list[i] == "WEP"):
        X_test.iloc[i, 6]= 28
    elif(list[i] == "STONOQ"):
        X_test.iloc[i, 6]= 29
    elif(list[i] == "AQ"):
        X_test.iloc[i, 6]= 30
    elif(list[i] == "LP"):
        X_test.iloc[i, 6]= 31
    else:
        X_test.iloc[i, 6] = -1






X_test.iloc[0:individuals, 4 ] =  X_test.iloc[0:individuals, 4 ] +X_test.iloc[0:individuals, 5 ]


X_test.iloc[0:individuals, 6 ]



### search by families



i =0
for line in X_test.iloc[0:individuals, 1]:
    if re.search("Mr.", line):
        X_test.iloc[i, 1] = 0

    if re.search("Miss.", line):
        X_test.iloc[i, 1] = 1
    if re.search("Master", line):
        X_test.iloc[i, 1] = 2
    if re.search("Don.", line):
        X_test.iloc[i, 1] = 3
    if re.search("Rev.", line):
        X_test.iloc[i, 1] = 4
    if re.search("Dr.", line):
        X_test.iloc[i, 1] = 5
    if re.search("Mme.", line):
        X_test.iloc[i, 1] = 6
    if re.search("Ms.", line):
        X_test.iloc[i, 1] = 7
    if re.search("Major.", line):
        X_test.iloc[i, 1] = 8
    if re.search("Mlle.", line):
        X_test.iloc[i, 1] = 9
    if re.search("Col.", line):
        X_test.iloc[i, 1] = 10
    if re.search("Capt.", line):
        X_test.iloc[i, 1] = 11
    if re.search("Countess.", line):
        X_test.iloc[i, 1] = 12
    if re.search("Jonkheer.", line):
        X_test.iloc[i, 1] = 13

    i = i + 1

X_test.iloc[0:individuals, 2] = X_test.iloc[0:individuals, 2].replace("male", 0)
X_test.iloc[0:individuals, 2] = X_test.iloc[0:individuals, 2].replace("female", 1)


X_test.iloc[0:individuals, 9] = X_test.iloc[0:individuals, 9].replace("S", 0)
X_test.iloc[0:individuals, 9] = X_test.iloc[0:individuals, 9].replace("Q", 1)
X_test.iloc[0:individuals, 9] = X_test.iloc[0:individuals, 9].replace("C", 2)

#Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked


X_1_teste = X_test.iloc[0:individuals, [0,  2,  6, 7, 10] ]


#for colunm in X_1_teste:
#    X_1_teste = X_1_teste.replace("NaN", X_1.mean())

X_1_teste = X_1_teste.interpolate()

X_1_teste = X_1_teste.astype(int)

##### fim teste ####

SVMClassifierSFFS(X_1, X_1_teste,  y.values)


randonForest(X_1, X_1_teste, y.values)


naiveBayes(X_1, X_1_teste, y.values)


def SVMClassifierSFFS(X, teste,  y):

    acc_media = 0
    precision_media = 0
    recall_media = 0
    auc_media = 0
    specificityMedia = 0
    sensibilityMedia = 0
    precisionMedia = 0
    accMedia = 0

    i = 0
    kf = KFold(n_splits=10, random_state=5, shuffle=True)
    # SVM
    clfSVM = svm.SVC(gamma=0.5)

    for train_index, test_index in kf.split(X):
        i = i + 1
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]

        selected_features_train = X_train
        selected_features_test = X_test

        clfSVM.fit(selected_features_train, y_train)
        y_predictSVM = clfSVM.predict(selected_features_test)

        y_predictTESTE = clfSVM.predict(teste)

        accSVM = accuracy_score(y_test, y_predictSVM)
        acc_media = acc_media + accSVM

        cm = confusion_matrix(y_test, y_predictSVM)
        print(cm)

        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]

        if ((tn + fp) != 0):
            specificity  = float(tn/ (tn + fp))
            specificityMedia = specificity + specificityMedia
        #
        if((tp + fn) !=0):
            sensibilityNew = float(tp/ (tp + fn))
            sensibilityMedia = sensibilityNew + sensibilityMedia

        if( (tp+fp) != 0):
            precisionNew = float(tp/(tp+fp))
            precisionMedia = precisionMedia + precisionNew

            precision, sensibility, fscore, support = precision_recall_fscore_support(y_test, y_predictSVM,average='binary')
            precision_media = precision_media + precision
            recall_media = recall_media + sensibility

        accNew =  float((tp+tn)/(tp+fn+fp+tn))
        accMedia = accMedia +accNew

        auc = roc_auc_score(y_test, y_predictSVM)
        auc_media = auc_media + auc

    print(y_predictTESTE)

    print("ACC_media: ", acc_media / 10)
    print("sensibilidade_media", recall_media/10)
    print("precision_media*: ", precision_media / 10, "\n", "recallNN_media*: ", recall_media / 10, "\n")
    print("roc auc_media*: ", auc_media / 10)

    print("especificidade", specificityMedia/10)
    print("sensibility", sensibilityMedia/10)
    print("precision", precisionMedia/10)
    print("accuracy", accMedia/10)
    return auc_media/10





def randonForest(X,teste,  y):

    acc_media = 0
    precision_media = 0
    recall_media = 0
    auc_media = 0
    specificityMedia = 0
    sensibilityMedia = 0
    precisionMedia = 0
    accMedia = 0

    i = 0

    model = RandomForestClassifier(n_estimators=1500,n_jobs=5)

    kf = KFold(n_splits=10, random_state=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model using the training sets and check score

        model.fit(X_train, y_train)
        # Predict Output
        predicted = model.predict(X_test)

        y_predictTESTE = model.predict(teste)

        # results
        cm = confusion_matrix(y_test, predicted)
        print(cm)

        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        #
        if ((tn + fp) != 0):
            specificity = float(tn / (tn + fp))
            specificityMedia = specificity + specificityMedia
        #
        if ((tp + fn) != 0):
            sensibilityNew = float(tp / (tp + fn))
            sensibilityMedia = sensibilityNew + sensibilityMedia

        if ((tp + fp) != 0):
            precisionNew = float(tp / (tp + fp))
            precisionMedia = precisionMedia + precisionNew

            precision, sensibility, fscore, support = precision_recall_fscore_support(y_test, predicted,
                                                                                      average='binary')
            precision_media = precision_media + precision
            recall_media = recall_media + sensibility

        accNew = float((tp + tn) / (tp + fn + fp + tn))
        accMedia = accMedia + accNew
        auc = roc_auc_score(y_test, predicted)
        auc_media = auc_media + auc

    print(y_predictTESTE)
    print("ACC_media: ", acc_media / 10)
    print("sensibilidade_media", recall_media / 10)
    print("precision_media*: ", precision_media / 10, "\n", "recallNN_media*: ", recall_media / 10, "\n")
    # print("fpr_media*: ", fpr_media, "\n", "tprNN_media", tpr_media, "\n")
    print("roc auc_media*: ", auc_media / 10)

    print("especificidade", specificityMedia / 10)
    print("sensibility", sensibilityMedia / 10)
    print("precision", precisionMedia / 10)
    print("accuracy", accMedia / 10)
    return auc_media / 10



from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def naiveBayes(X,teste,  y):


    acc_media = 0
    precision_media = 0
    recall_media = 0
    auc_media = 0
    specificityMedia = 0
    sensibilityMedia = 0
    precisionMedia = 0
    accMedia = 0

    gnb = MultinomialNB ()

    kf = KFold(n_splits=10, random_state=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]

        model = gnb.fit(X_train, y_train)
        predicted = model.predict(X_test)

        y_predictTESTE = model.predict(teste)

        # results
        cm = confusion_matrix(y_test, predicted)
        print(cm)

        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]

        if ((tn + fp) != 0):
            specificity = float(tn / (tn + fp))
            specificityMedia = specificity + specificityMedia
        #
        if ((tp + fn) != 0):
            sensibilityNew = float(tp / (tp + fn))
            sensibilityMedia = sensibilityNew + sensibilityMedia

        if ((tp + fp) != 0):
            precisionNew = float(tp / (tp + fp))
            precisionMedia = precisionMedia + precisionNew

            precision, sensibility, fscore, support = precision_recall_fscore_support(y_test, predicted,
                                                                                      average='binary')
            precision_media = precision_media + precision
            recall_media = recall_media + sensibility

        accNew = float((tp + tn) / (tp + fn + fp + tn))
        accMedia = accMedia + accNew

        auc = roc_auc_score(y_test, predicted)
        auc_media = auc_media + auc

    print(y_predictTESTE)
    print("ACC_media: ", acc_media / 10)
    print("sensibilidade_media", recall_media / 10)
    print("precision_media*: ", precision_media / 10, "\n", "recallNN_media*: ", recall_media / 10, "\n")
    print("roc auc_media*: ", auc_media / 10)

    print("especificidade", specificityMedia / 10)
    print("sensibility", sensibilityMedia / 10)
    print("precision", precisionMedia / 10)
    print("accuracy", accMedia / 10)
    return auc_media / 10