
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np


# ### Leemos el archivo con la máxima extensión, luego ya reduciremos la memoria

# In[4]:

data_csv = pd.read_csv('loan.csv',low_memory=False)


# # Preparacion de datos
# Vamos a preparar los datos para entrenar después un modelo clasificatorio
# ## Comprobamos que no hay duplicidadaes

# In[5]:

print(len(data_csv))
print(len(data_csv.duplicated()))


# ## Quitamos variables que no deben tener correlación lógica con el resultado de la clasificación

# In[6]:

del data_csv['emp_title']
del data_csv['desc']
del data_csv['id']
del data_csv['member_id']
del data_csv['purpose']
del data_csv['title']
del data_csv['url']


# ## Quitamos variables que no contienen suficiente información

# In[7]:

del data_csv['annual_inc_joint']
del data_csv['dti_joint']
del data_csv['verification_status_joint']


# ## Modificación de variables. 
# Si están informadas es malo, ya que se refieren a hechos delinquivos. Al referirse en número, damos un valor 0 para que el modelo aprenda que los que tienen un número mayor son malos.

# In[8]:

data_csv.mths_since_last_delinq.fillna(0,inplace=True)
data_csv.mths_since_last_record.fillna(0,inplace=True)
data_csv.mths_since_last_major_derog.fillna(0,inplace=True)


# ## Eliminación de variables. 
# Con información insuficiente o que hacen referencia a información a posteriori

# In[9]:

del data_csv['il_util']
del data_csv['max_bal_bc']
del data_csv['all_util']


# In[10]:

data_csv = data_csv.drop(['recoveries', 'collection_recovery_fee', 'out_prncp','out_prncp_inv',
            'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
            'last_pymnt_d','last_pymnt_amnt','total_rec_late_fee',
            'funded_amnt','funded_amnt_inv'], axis=1)


# ## Creación de variables
# Creamos una variable relacionada con el hecho de haber habierto una cuenta o de tener una *il*

# In[11]:

data_csv['il']=0
data_csv.il[data_csv[['open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il']].isnull().T.apply(any)] = 1


# In[12]:

del data_csv['open_acc_6m']
del data_csv['open_il_6m']
del data_csv['open_il_12m']
del data_csv['open_il_24m']
del data_csv['mths_since_rcnt_il']
del data_csv['total_bal_il']


# Creamos una variable binaria realacionada con el hecho de tener un producto revolving

# In[13]:

data_csv['rv']=0
data_csv.rv[data_csv[['open_rv_12m','open_rv_24m']].isnull().T.apply(any)] = 1


# In[14]:

del data_csv['open_rv_12m']
del data_csv['open_rv_24m']


# Creamos una variable binaria relacionada con el hecho de tener una inquery

# In[15]:

data_csv['iq']=0
data_csv.iq[data_csv[['inq_fi','total_cu_tl','inq_last_12m']].isnull().T.apply(any)] = 1


# In[16]:

del data_csv['inq_fi']
del data_csv['total_cu_tl']
del data_csv['inq_last_12m']


# In[247]:

data2 = data_csv.copy()


# Ahora ya eliminamos los registros con algún missing de forma definitiva

# In[17]:

data_csv = data_csv.dropna()


# ## Quitamos Outliers
# Vamos a quitar los Outliers revisando variable a variable como están distribuidas teniendo en cuenta que después vamos a hacer un escalado y los máximos muy alejados estropearian dicho escalado

# In[18]:

data_csv.annual_inc[data_csv['annual_inc'] > 600000] = 600000
data_csv.dti[data_csv['dti']>40]=40
data_csv.delinq_2yrs[data_csv['delinq_2yrs']>8]=8
data_csv.inq_last_6mths[data_csv['inq_last_6mths']>8]=8
data_csv.mths_since_last_delinq[data_csv['mths_since_last_delinq']>100]=100
data_csv.open_acc[data_csv['open_acc']>50]=50
data_csv.pub_rec[data_csv['pub_rec']>5]=5
data_csv.revol_bal[data_csv['revol_bal']>80000]=80000
data_csv.revol_util[data_csv['revol_util']>120]=120
data_csv.total_acc[data_csv['total_acc']>100]=100
data_csv.collections_12_mths_ex_med[data_csv['collections_12_mths_ex_med']>3]=3
data_csv.mths_since_last_major_derog[data_csv['mths_since_last_major_derog']>120]=120
del data_csv['policy_code']


# ## Hacemos variables OneHot
# Pero solamente de las variables que no son ordinales, es decir que no existe una diferencia real entre el primer y el segundo nivel. Ordinales son: term, emp_length, grade, sub_grade, issue_d, earliest_cr_line, last_credit_pull_d. Aunque las tres últimas necesitarán ser ordenadas para que tenga sentido que sean ordinales. Además, vamos a eliminar la variable *zip_code* por ser bastante repetitiva con *addr_state* y crearía un número de columnas dummies demasiado grande

# Vamos a utilizar el pd.get_dummies. Aunque la función de numpy OneHotEncoder se podría introducir en el pipeline directamente. Así se simplifica la gestión de los datos.

# In[19]:

dummies = pd.get_dummies(data_csv[['home_ownership','verification_status','pymnt_plan','addr_state','initial_list_status','application_type']])


# In[20]:

del data_csv['next_pymnt_d']
del data_csv['home_ownership']
del data_csv['verification_status']
del data_csv['pymnt_plan']
del data_csv['zip_code']
del data_csv['addr_state']
del data_csv['initial_list_status']
del data_csv['application_type']


# Variables que son de fecha, al hacerlas enteras se volverán directamente números enteros

# In[21]:

data_csv.issue_d = pd.to_datetime(data_csv.issue_d).values.astype('int')
data_csv.earliest_cr_line = pd.to_datetime(data_csv.earliest_cr_line).values.astype('int')
data_csv.last_credit_pull_d = pd.to_datetime(data_csv.last_credit_pull_d).values.astype('int')


# Vamos a mapear el resto de variables

# In[22]:

from sklearn.preprocessing import LabelEncoder

sub_grade = LabelEncoder()
sub_grade.fit(np.unique(data_csv.sub_grade))
data_csv.sub_grade = sub_grade.transform(data_csv.sub_grade)


# In[23]:

grade = LabelEncoder()
grade.fit(np.unique(data_csv.grade))
data_csv.grade = grade.transform(data_csv.grade)


# In[24]:

term = LabelEncoder()
term.fit(np.unique(data_csv.term))
data_csv.term = term.transform(data_csv.term)


# In[25]:

emp_length_mapping = {
"< 1 year": 0.0,
"1 year": 1.0,
"2 years": 2.0,
"3 years": 3.0,
"4 years": 4.0,
"5 years": 5.0,
"6 years": 6.0,
"7 years": 7.0,
"8 years": 8.0,
"9 year": 9.0,
"10+ years": 10.0,
"n/a": 5.5
}
data_csv.emp_length = data_csv.emp_length.map(emp_length_mapping)


# ## VAMOS A CONCATENAR LAS DUMMIES Y EL RESTO PARA TENER EL TABLÓN DEFINITIVO PARA ENTRERNAR

# In[26]:

data = pd.concat([data_csv,dummies],axis=1).dropna()


# ## VAMOS A CATEGORIZAR LOAN_STATUS
# No es la forma más elegante pero funciona

# In[27]:

label = data['loan_status']


# In[28]:

y=[]
for x in label:
    if x in ['Current','Fully Paid','Issued']:
        y.append(1)
    else:
        y.append(0)


# In[29]:

from collections import Counter
Counter(y)


# # Empezemos con sklearn

# In[30]:

del data['loan_status']


# ## Le metemos toda la información escalada a un Naive Bayes

# In[31]:

x = data.dropna()


# In[32]:

x = np.matrix(x)


# In[33]:

x.shape


# In[34]:

from sklearn.preprocessing import MinMaxScaler


# In[35]:

escalador = MinMaxScaler()


# In[36]:

X = escalador.fit_transform(x)


# In[37]:

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# In[38]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[60]:

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred_prob = gnb.fit(X_train, y_train).predict_proba(X_test)


# In[61]:

from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test,y_pred))


# In[64]:

classification_report(y_test, y_pred)


# In[41]:

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[42]:

auc_1 = roc_auc_score(y_train,pd.DataFrame(y_pred_prob)[1])
print(auc_1)


# In[44]:

import matplotlib.pyplot as plt
def plot_roc(Y, Y_scores):
    fpr, tpr, thresholds = roc_curve(Y, Y_scores, pos_label = 1)
    plt.figure(1, figsize=(6,6))
    plt.xlabel('Tasa falsos positivos')
    plt.ylabel('Tasa verdaderos positivos')
    plt.title('Curva ROC')
    plt.plot(fpr, tpr)
    plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')


# In[55]:

get_ipython().magic('matplotlib inline')


# In[57]:

plot_roc(y_train,pd.DataFrame(y_pred_prob)[1])

