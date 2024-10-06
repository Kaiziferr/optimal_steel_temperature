from unicodedata import normalize

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def porcentaje_nulos_x_caracteristica(data:pd.DataFrame, **kwargs)->None:
  """Calcular el porcentaje de valores nulos por caracteristicas"""
  data.isnull().melt().pipe(
        lambda df: (
            sns.displot(
                data=df,
                y='variable',
                hue='value',
                multiple='fill',
                aspect=2
            ).set(**kwargs)
        )
    )
  
def null_features_per_record(data, figsize=(9,8), **kwargs):
  """Identifica el número de caracteristicas nulas por registro"""
  plt.figure(figsize=figsize)
  (
      data
      .isnull()
      .transpose()
      .pipe(
          lambda df: (
              sns.heatmap(
                  data = df
                  
              ).set(**kwargs)
          )
      )
  )

def normalize_word(word)->str:
    """Normaliza palabras"""
    word = word.replace(' ', '_')
    find_guion = word.find('_')
    list_word = []
    if find_guion:
        list_word = [w for w in word.split('_') if w != '']
    else:
        list_word = word
    word = list(map(lambda x: x.lower(), list_word))
    word = [normalize('NFKD', c).encode('ASCII', 'ignore').decode() for c in word]
    word = "_".join(word)
    return word

def normalize_name_columns(columns):
  """Normaliza columnas"""
  columns = list(map(lambda x: normalize_word(x), columns))
  return columns

def calculate_time(
        data:list,
        name_start_f: str,
        name_end_f: str)->float:
    """Calculate time in fractions"""
    try:
        seconds_start =  np.datetime64(data[name_start_f]) 
        seconds_end =  np.datetime64(data[name_end_f])
        seconds = seconds_end - seconds_start
        seconds = round(seconds / np.timedelta64(1, 'm'), 3)
        return seconds         
    except Exception as e:
        print(repr(e))
        return 0


def calculate_time(
        data:list,
        name_start_f: str,
        name_end_f: str)->float:
    """Calculate time in fractions"""
    try:
        seconds_start =  np.datetime64(data[name_start_f]) 
        seconds_end =  np.datetime64(data[name_end_f])
        seconds = seconds_end - seconds_start
        seconds = round(seconds / np.timedelta64(1, 'm'), 3)
        return seconds         
    except Exception as e:
        print(repr(e))
        return 0
    

def score_mi_plot(score, features):
    """Plotting MI score"""
    plt.hlines(y=features, xmin=0, xmax=score, color='skyblue')
    plt.plot(score, features, "o")
    plt.title("Mutual Information Scores", loc='left')
    plt.xlabel('Score')
    plt.ylabel('Feature Group')



def componentes_principales(data:object,
                            min_explained_variance:int=0.85,
                            min_component:int=3,
                            **kwards)->tuple:
  """simple method of PCA, that return data transform"""
  final_comp = 0
  for comp in range(min_component, data.shape[1]):
    pca = PCA(n_components=comp, **kwards)
    pca.fit(data)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() >= min_explained_variance:
      break

  final_pca = PCA(n_components=final_comp, **kwards)
  final_pca.fit(data)
  data_df = final_pca.transform(data)
  info = "Using {} components, we can explain {}% of the variability\
  in the original data.".format(final_comp,comp_check.sum())

  return data_df, final_pca, info


def scree_plot(
    final_pca:object,
    paleta,
    size:tuple=(10,5),
    title:str='Scree Plot'
    )->None:
  """Permite visualizar la varianza de cada componente"""
  number_components = final_pca.n_components
  principal_components = [f'PC{i+1}' for i in range(number_components)]
  explained_variance = final_pca.explained_variance_ratio_
  cumulative_variance = np.cumsum(explained_variance)
  plt.figure(figsize = size)
  plt.bar(principal_components, explained_variance, color=paleta[0])
  plt.plot(
      principal_components,
      cumulative_variance,
      'o-',
      linewidth=2,
      color=paleta[1])

  for i,j in zip(principal_components, cumulative_variance):
    plt.annotate(str(round(j,2)), xy=(i, j))
  plt.title(title)
  plt.xlabel('Principal Components')
  plt.ylabel('Explained Variance')
  plt.show()