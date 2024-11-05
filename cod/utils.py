# utils.py
import pandas as pd
from beacons import Point, check_intersection, beacons  
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def numera_direction(direction_string):
  if direction_string == 'norte':
      return 1
  elif direction_string == 'leste':
      return 2
  elif direction_string == 'sul':
      return 3
  elif direction_string == 'oeste':
      return 4
  
def inserir_linha(idx, df, df_inserir):
    dfA = df.iloc[:idx]
    dfB = df.iloc[idx:]

    if df_inserir.empty:
        return pd.concat([dfA, dfB]).reset_index(drop=True)
    else:
        return pd.concat([dfA, df_inserir, dfB]).reset_index(drop=True)

def separa_sub_df(df):
    df_intercepta = pd.DataFrame(columns=df.columns)
    df_nao_intercepta = pd.DataFrame(columns=df.columns)

    lista = list(df.columns)
    for index, row in df.iterrows():
        row_df = pd.DataFrame([row])
        if check_intersection(Point(row['X'], row['Y']), numera_direction(row['Direction']), beacons[lista[0]]):
            df_intercepta = pd.concat([df_intercepta, row_df], ignore_index=True)
        else:
            df_nao_intercepta = pd.concat([df_nao_intercepta, row_df], ignore_index=True)

    min_len = min(len(df_intercepta), len(df_nao_intercepta))
    df_intercepta = df_intercepta.iloc[:min_len]
    df_nao_intercepta = df_nao_intercepta.iloc[:min_len]

    return df_intercepta, df_nao_intercepta

def verifica_serie(serie):
    lista = list(serie.index)
    x = serie['X']
    y = serie['Y']
    direction = serie['Direction']
    direcao_numero = numera_direction(direction) 
    ponto = Point(x, y)
    return check_intersection(ponto, direcao_numero, beacons[lista[0]])


def process_row(row, beacons, direction=None):
    programmer_position = Point(row['X'], row['Y'])
    direction = numera_direction(direction if direction else row['Direction'])

    results = {}
    for beacon_name in beacons:
      if beacon_name in row.index:
          beacon_position = beacons[beacon_name]
          intersects = check_intersection(programmer_position, direction, beacon_position)
          results[beacon_name] = 1 if intersects else 0

    return results


#testando
if __name__ == "__main__":
    df = pd.read_csv("./resultados-all-beacons/train_df_norte.csv")

    
    B2_train_model = df[['B7_min', 'X', 'Y', 'Direction']]

    B2_train_model_intercepta, B2_train_model_nao_intercepta = separa_sub_df(B2_train_model)

    print(f"Train: Intercepta: {len(B2_train_model_intercepta)}, Não Intercepta: {len(B2_train_model_nao_intercepta)}")
    



