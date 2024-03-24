from unicodedata import normalize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
