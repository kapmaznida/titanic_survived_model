import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helpers.eda import *
from helpers.data_prep import *

from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    df = pd.read_csv("weeks/week_06/titanic.csv")
    return df

data = load()
check_df(data)

data["Age"].hist(bins = 5)
plt.show()

# categoric, numeric, categoric but cardinal değerlerimizi analiz edelim.

cat_cols, num_cols, cat_but_car_cols = grab_col_names(data, cat_th=10, car_th=20)
# Observations: 891
# Variables: 12
# cat_cols: 6
# num_cols: 3
# cat_but_car: 3
# num_but_cat: 4

for col in cat_cols:
    target_summary_with_cat(data, "Survived", col)
# Dikkatimi çeken kategorik sütunlar:
#         TARGET_MEAN
# Sex
# female        0.742
# male          0.189

#         TARGET_MEAN
# Pclass
# 1             0.630
# 2             0.473
# 3             0.242

for col in num_cols:
    target_summary_with_num(data, "Survived", col)

# PassengerId Id olmasından mütevellit saöma sapan geldi anlamsız çöp.
#            Fare
# Survived
# 0        22.118
# 1        48.395
# Daha çok para ödeyenler yaşamış çıkarımını yapabilecek gibiyiz.

#             Age
# Survived
# 0        30.626
# 1        28.344
# Yaş konusunda can alıcı bir fark bulamadık.

for col in cat_cols:
    cat_summary(data, col, plot=True)

for col in num_cols:
    num_summary(data, col, plot=False)


def titanic_df_pred(df):
    dataframe = df.copy()
    # Feature Engineering
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    dataframe["NEW_TITLE"] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    dataframe["NEW_IS_ALONE"] = ["YES" if i > 1 else "NO" for i in dataframe["FAMILY_SIZE"]]
    cat_cols, num_cols, cat_but_car_cols = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # Missing Values
    missing_cols = [col for col in dataframe.columns if (dataframe[col].isnull().any()) & (col != "Cabin")]
    for i in missing_cols:
        if i == "AGE":
            dataframe[i].fillna(dataframe.groupby("PCLASS")[i].transform("median"),inplace=True)
        elif dataframe[i].dtype == "O":
            dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
        else:
            dataframe[i].fillna(dataframe[i].media(), inplace=True)

    # Outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    deleted_cols = ["CABIN", "SIBSP", "PARCH", "TICKET", "NAME"]
    dataframe = dataframe.drop(deleted_cols, axis=1)

    dataframe["NEW_AGE_CAT"] = pd.cut(dataframe["AGE"], bins=[0, 25, 40, 55, dataframe["AGE"].max()+1],
                                      labels=[1, 2, 3, 4]).astype('int')

    dataframe.loc[(dataframe["SEX"] == "male") & (dataframe["AGE"] <= 25), "NEW_SEX_CAT"] = "youngmale"
    dataframe.loc[(dataframe["SEX"] == "male") & (
        dataframe["AGE"] > 25) & (dataframe["AGE"] < 55), "NEW_SEX_CAT"] = "maturemale"
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 20) & (dataframe['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car_cols = grab_col_names(dataframe)

    # Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    # One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, cat_cols, True)

    # Standart Scaler
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe

data.head()
data.describe()

data_prep = titanic_df_pred(data)

data_prep.head()
data_prep.describe()
check_df(data_prep)

data_prep['SURVIVED_1'].head()

######################################################
# Model & Prediction
######################################################

# Bağımlı ve bağımsız değişkelerin seçilmesi:
y = data_prep["SURVIVED_1"]
X = data_prep.drop(["PASSENGERID", "SURVIVED_1"], axis=1)


# Model:
log_model = LogisticRegression().fit(X, y)

log_model.intercept_
#array([2.38589185])
log_model.coef_
# array([[-0.51430093,  0.13614893, -0.87138575,  0.01613539, -0.45085799,
#         -0.55723596, -2.19442671,  0.13638717, -1.38351177, -1.1180432 ,
#         -0.1262332 ,  0.37703828, -0.20358805, -0.24457142, -0.36497517,
#         -1.27703178,  0.90344417,  0.74096517,  0.93320813,  0.85067312,
#         -0.98068994, -1.28774465, -0.15679307, -1.21766196,  0.37812354,
#          0.37154525,  0.17345023]])

# Tahmin
y_pred = log_model.predict(X)

y_pred[0:10]
#  array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=uint8)
y[0:10]


######################################################
# Model Evaluation
######################################################


# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)


# Başarı skorları:
print(classification_report(y, y_pred))
#               precision    recall  f1-score   support
#            0       0.85      0.88      0.87       549
#            1       0.80      0.75      0.77       342
#     accuracy                           0.83       891
#    macro avg       0.82      0.82      0.82       891
# weighted avg       0.83      0.83      0.83       891


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
#  0.8826681153399589



######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = data_prep["SURVIVED_1"]
X = data_prep.drop(["PASSENGERID", "SURVIVED_1"], axis=1)

log_model = LogisticRegression().fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



cv_results['test_accuracy'].mean()
# Accuracy:  0.8215303496327915

cv_results['test_precision'].mean()
# Precision:  0.7780146884624497

cv_results['test_recall'].mean()
# Recall: 0.7511935208866155

cv_results['test_f1'].mean()
# F1-score: 0.7630113911942642

cv_results['test_roc_auc'].mean()
# AUC: 0.8696081684296448


######################################################
# Prediction for A New Observation
#####################################################

X.columns

random_user = X.sample(1, random_state=42)

log_model.predict(random_user)

