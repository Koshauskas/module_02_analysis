import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from itertools import combinations

from scipy.stats import ttest_ind

df = pd.read_csv('stud_math.xls')

print(df.head(10))

print(df.info())

def fill_none(row):
    df[row] = df[row].astype(str).apply(lambda x: None if x.strip() == '' else x)
    df[row] = df[row].apply(lambda x: float(x))


print(df.school.nunique())#количество уникальных значений в столбце
print(df.school.value_counts())#подсчет количества каждого из уникальных значений
#так как в столбце нет пропусков, переходим к следующему столбцу

print(df.sex.nunique())
print(df.sex.value_counts())
#так как в столбце нет пропусков, переходим к следующему столбцу

print(df.age.describe())

df.age.hist()#диаграмма значений возраста
plt.show()
#так как в столбце нет пропусков, переходим к следующему столбцу

print(df.address.nunique())
print(df.address.value_counts())
df.address = df.address.astype(str).apply(lambda x: None if x.strip() == '' else x)#заменяем пропущенные значения

print(df.famsize.nunique())
print(df.famsize.value_counts())
df.famsize = df.famsize.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.Pstatus.nunique())
print(df.Pstatus.value_counts())
df.Pstatus = df.Pstatus.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.Medu.nunique())
print(df.Medu.value_counts())
fill_none('Medu')


print(df.Fedu.nunique())
print(df.Fedu.value_counts())
fill_none('Fedu')
df = df.loc[df.Fedu.between(0.0, 4.0)]
print(df.Fedu.value_counts())

print(df.Mjob.nunique())
print(df.Mjob.value_counts())
df.Mjob = df.Mjob.astype(str).apply(lambda x: None if x.strip() == '' else x)


print(df.Fjob.nunique())
print(df.Fjob.value_counts())
df.Fjob = df.Fjob.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.reason.nunique())
print(df.reason.value_counts())
df.reason = df.reason.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.guardian.nunique())
print(df.guardian.value_counts())
df.guardian = df.guardian.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.traveltime.nunique())
print(df.traveltime.value_counts())
fill_none('traveltime')

print(df.studytime.nunique())
print(df.studytime.value_counts())
fill_none('studytime')

print(df.failures.nunique())
print(df.failures.value_counts())
fill_none('failures')

print(df.schoolsup.nunique())
print(df.schoolsup.value_counts())
df.schoolsup = df.schoolsup.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.famsup.nunique())
print(df.famsup.value_counts())
df.famsup = df.famsup.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.paid.nunique())
print(df.paid.value_counts())
df.paid = df.paid.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.activities.nunique())
print(df.activities.value_counts())
df.activities = df.activities.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.nursery.nunique())
print(df.nursery.value_counts())
df.nursery = df.nursery.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df['studytime, granular'].nunique())
print(df['studytime, granular'].value_counts())
del df['studytime, granular']


print(df.higher.nunique())
print(df.higher.value_counts())
df.higher = df.higher.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.internet.nunique())
print(df.internet.value_counts())
df.internet = df.internet.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.romantic.nunique())
print(df.romantic.value_counts())
df.romantic = df.romantic.astype(str).apply(lambda x: None if x.strip() == '' else x)

print(df.famrel.nunique())
print(df.famrel.value_counts())
fill_none('famrel')
df = df.loc[df.famrel.between(0.0, 5.0)]
print(df.famrel.value_counts())

print(df.freetime.nunique())
print(df.freetime.value_counts())
fill_none('freetime')

print(df.goout.nunique())
print(df.goout.value_counts())
fill_none('goout')

print(df.health.nunique())
print(df.health.value_counts())
fill_none('health')

print(df.absences.nunique())
print(df.absences.value_counts())
fill_none('absences')

print(df.score.nunique())
print(df.score.value_counts())
fill_none('score')
print(df.isna().sum())

print(df.corr())

def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='score',
                data=df.loc[df.loc[:, column].isin(df.loc[:, column].value_counts().index[:10])],
               ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()

for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic' ]:
    get_boxplot(col)

def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'],
                        df.loc[df.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break

for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
    get_stat_dif(col)
    
df_for_model = df.loc[:,['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout','health', 'absences', 'higher']]