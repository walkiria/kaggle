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




def SVMClassifierSFFS(X, y):

    acc_media = 0
    precision_media = 0
    recall_media = 0
    fpr_media = 0
    tpr_media = 0
    auc_media = 0
    specificityMedia = 0
    sensibilityMedia = 0
    precisionMedia = 0
    accMedia = 0
    specificity = 0
    sensibilityNew = 0
    i = 0
    kf = KFold(n_splits=10, random_state=5, shuffle=True)
    # SVM
    clfSVM = svm.SVC()

    for train_index, test_index in kf.split(X):
        i = i + 1
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]

        selected_features_train = X_train
        selected_features_test = X_test

        clfSVM.fit(selected_features_train, y_train)
        y_predictSVM = clfSVM.predict(selected_features_test)

        accSVM = accuracy_score(y_test, y_predictSVM)
        acc_media = acc_media + accSVM



        fprSVM, tprSVM, thresholdsSVM = roc_curve(y_test, y_predictSVM)

        cm = confusion_matrix(y_test, y_predictSVM)
        print(cm)

        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        #
        # print
        # tp
        # print
        # tn
        #
        #
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
        #fpr_media = fprSVM + fpr_media
        #tpr_media = tpr_media + tprSVM

        auc = roc_auc_score(y_test, y_predictSVM)
        auc_media = auc_media + auc

    print("ACC_media: ", acc_media / 10)
    print("sensibilidade_media", recall_media/10)
    print("precision_media*: ", precision_media / 10, "\n", "recallNN_media*: ", recall_media / 10, "\n")
    #print("fpr_media*: ", fpr_media, "\n", "tprNN_media", tpr_media, "\n")
    print("roc auc_media*: ", auc_media / 10)

    print("especificidade", specificityMedia/10)
    print("sensibility", sensibilityMedia/10)
    print("precision", precisionMedia/10)
    print("accuracy", accMedia/10)
    return auc_media/10



def randonForest(X, y):

    acc_media = 0
    precision_media = 0
    recall_media = 0
    fpr_media = 0
    tpr_media = 0
    auc_media = 0
    specificityMedia = 0
    sensibilityMedia = 0
    precisionMedia = 0
    accMedia = 0
    specificity = 0
    sensibilityNew = 0
    i = 0

    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Random Forest object
    model = RandomForestClassifier(n_estimators=1000)

    kf = KFold(n_splits=10, random_state=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model using the training sets and check score

        model.fit(X_train, y_train)
        # Predict Output
        predicted = model.predict(X_test)

        # results
        cm = confusion_matrix(y_test, predicted)
        print(cm)

        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        #
        # print
        # tp
        # print
        # tn
        #
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
        # fpr_media = fprSVM + fpr_media
        # tpr_media = tpr_media + tprSVM

        auc = roc_auc_score(y_test, predicted)
        auc_media = auc_media + auc

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




def NNClassifier(X, y):
    acc_media = 0
    precision_media = 0
    recall_media = 0
    fpr_media = 0
    tpr_media = 0
    auc_media = 0
    i = 0
    kf = KFold(n_splits=10, random_state=5, shuffle=True)

    idx = fisherScore(X, y, num_fea)


    for train_index, test_index in kf.split(X):
        i = i + 1
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y[train_index], y[test_index]

        n_samplesTrain, n_featuresTrain = np.shape(X_train)
        n_samplesTest, n_featuresTest = np.shape(X_test)

        clfNN.fit(X_train, y_train)
        y_predictNN = clfNN.predict(X_test)
        accNN = accuracy_score(y_test, y_predictNN)

        acc_media = acc_media + accNN

        precision, sensibility, fscore, support = precision_recall_fscore_support(y_test, y_predictNN,
                                                                                  average='binary')
        precision_media = precision_media + precision
        recall_media = recall_media + sensibility

        fpr, tpr, thresholds = roc_curve(y_test, y_predictNN)

        fpr_media = fpr_media + fpr
        tpr_media = tpr_media + tpr

        auc = roc_auc_score(y_test, y_predictNN)
        auc_media = auc_media + auc

    print("ACC_media: ", acc_media / 10)
    print("precision_media*: ", precision_media / 10, "\n", "recallNN_media*: ", recall_media / 10, "\n")
    # print("fprNN_media*: ", fpr_media/10, "\n", "tprNN_media", tpr_media/10, "\n")
    print("roc auc_media*: ", auc_media / 10)

    return auc_media


