from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
import os

from utils import separa_sub_df, process_row
from beacons import beacons

class Generator:
    def __init__(self, train_df, model_type='LinearRegression'):
        self.train_df = train_df
        self.beacons = beacons
        self.model_type = model_type
        self.model_com_obs = None
        self.model_sem_obs = None
        self.__initialize_models()

    def __initialize_models(self):
        self.model_com_obs = self.train_models(com_obs=True)
        self.model_sem_obs = self.train_models(com_obs=False)

    def train_models(self, com_obs=True):
        nome_beacon = 'B1_min'
        train_model = self.train_df[['X', 'Y', 'RP', nome_beacon, 'Direction']]

        train_model_norte = train_model[train_model['Direction'] == 'norte'].copy()
        train_model_sul = train_model[train_model['Direction'] == 'sul'].copy()

        dados_alinhados = []

        todos_rps = set(train_model_sul['RP'])
        
        for rp in todos_rps:
            #print("Para RP, ", rp)
            train_model_norte_rp = train_model_norte[train_model_norte['RP'] == rp].sort_values(by='B1_min', ascending=False)
            train_model_sul_rp = train_model_sul[train_model_sul['RP'] == rp].sort_values(by='B1_min', ascending=False)

            norte_len = len(train_model_norte_rp)
            sul_len = len(train_model_sul_rp)


            min_length = min(norte_len, sul_len)

            for i in range(min_length):
                #print(f"Pega a Linha {i} do Norte_RP, com a linha {i} do Sul_RP")
                rssi_norte = train_model_norte_rp.iloc[i]['B1_min']
                rssi_sul = train_model_sul_rp.iloc[i]['B1_min']
                dados_alinhados.append((rssi_norte, rssi_sul, rp))  

        df_alinhado = pd.DataFrame(dados_alinhados, columns=['B1_min_norte', 'B1_min_sul', 'RP'])

        df_alinhado = df_alinhado.groupby('RP').agg(lambda x: x.mode().iloc[0])
        print(df_alinhado)

        if com_obs:
            X = df_alinhado[['B1_min_norte']].values
            y = df_alinhado[['B1_min_sul']].values.ravel()
        else:
            X = df_alinhado[['B1_min_sul']].values
            y = df_alinhado[['B1_min_norte']].values.ravel()

        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(f"Não há dados disponíveis para o beacon: {nome_beacon}")

        modelo = self.get_model_instance()
        modelo.fit(X, y)

        return modelo




    def get_model_instance(self):
        if self.model_type == 'LinearRegression':
            return LinearRegression()
        elif self.model_type == 'KNN':
            return KNeighborsRegressor(n_neighbors=5)
        elif self.model_type == 'DecisionTree':
            return DecisionTreeRegressor()
        elif self.model_type == 'SVR':
            return SVR(kernel='rbf')  # Kernel não linear
        elif self.model_type == 'RandomForest':
            return RandomForestRegressor()
        elif self.model_type == 'Ridge':
            return Ridge()
        elif self.model_type == 'Lasso':
            return Lasso()
        elif self.model_type == 'ElasticNet':
            return ElasticNet()
        elif self.model_type == 'GaussianProcess':
            return GaussianProcessRegressor()
        elif self.model_type == 'GradientBoosting':
            return GradientBoostingRegressor()
        elif self.model_type == 'MLP':
            return MLPRegressor(max_iter=1000)
        elif self.model_type == 'ExtraTrees':
            return ExtraTreesRegressor()
        elif self.model_type == 'HistGradientBoosting':
            return HistGradientBoostingRegressor()
        elif self.model_type == 'Huber':
            return HuberRegressor()
        elif self.model_type == 'Stacking':
            return StackingRegressor(estimators=[
                ('ridge', Ridge()),
                ('svr', SVR())
            ])
        elif self.model_type == 'MultiOutput':
            return MultiOutputRegressor(DecisionTreeRegressor())
        elif self.model_type == 'KernelRidge':
            return KernelRidge()
        elif self.model_type == 'RadiusNeighbors':
            return RadiusNeighborsRegressor(radius=1.0)
        elif self.model_type == 'ExtraTree':
            return ExtraTreeRegressor()
        elif self.model_type == 'AdaBoost':
            return AdaBoostRegressor()
        elif self.model_type == 'Bagging':
            return BaggingRegressor(base_estimator=DecisionTreeRegressor())
        elif self.model_type == 'Voting':
            return VotingRegressor(estimators=[
                ('linear', LinearRegression()),
                ('tree', DecisionTreeRegressor()),
                ('svr', SVR())
            ])
        elif self.model_type == 'Logistic':
            return LogisticRegression()
        
        else:
            raise ValueError(f"Modelo '{self.model_type}' não é suportado.")

    def predict_rssi(self, rssi_value, com_obs=True):
        model = self.model_com_obs if com_obs else self.model_sem_obs
        prediction = model.predict(np.array([[rssi_value]]))[0]
        #print("com_obs", com_obs)
        #print("RSSI data: ", rssi_value)
        #print("Predicted RSSI: ", prediction if not isinstance(prediction, np.ndarray) else prediction[0])
        return prediction if not isinstance(prediction, np.ndarray) else prediction[0]

    def gera_df(self, df, direction):
        # Cria uma cópia do DataFrame original 
        result_df = df.copy()

        # Processa cada linha do DataFrame
        for index, row in df.iterrows():
            #print("Processing row ", index)
            intercepta_result = process_row(row, self.beacons, direction=direction) 
            #print("Intercepted ", intercepta_result)
            intercepta_original = process_row(row, self.beacons)
            #print("Original Intercepted ", intercepta_original)

            for beacon_name, intercepta in intercepta_result.items():
                original_intercepta = intercepta_original[beacon_name]

                if intercepta == 1 and original_intercepta == 0:
                    # Modifica o valor usando com_obs=False
                    #print(f"Modifica o valor usando com_obs=False para o beacon_name {beacon_name}")
                    rssi_value = self.predict_rssi(row[beacon_name], com_obs=False)
                elif intercepta == 0 and original_intercepta == 1:
                    # Modifica o valor usando com_obs=True
                    #print(f"Modifica o valor usando com_obs=True para o beacon_name {beacon_name}")
                    rssi_value = self.predict_rssi(row[beacon_name], com_obs=True)
                else:
                    # Caso contrário, mantenha o valor original
                    rssi_value = row.get(beacon_name, np.nan)

                # Atualiza o DataFrame com o novo valor
                result_df.at[index, beacon_name] = rssi_value

        return result_df


if __name__ == '__main__':
    df = pd.read_csv("./dataset.csv")
    
    train_df = pd.read_csv("./resultados-all-beacons/train_df.csv")

    train_df_norte = pd.read_csv("./resultados-all-beacons/train_df_norte.csv")
    
    # Filtra o DataFrame para manter apenas as linhas onde a coluna 'RP' termina com '2'
    train_df_rp_2 = train_df[train_df['RP'].str.endswith('2')]
 

    models = ['LinearRegression', 'KNN', 'DecisionTree', 'SVR', 
          'RandomForest', 'Ridge', 'Lasso', 'ElasticNet', 
          'MLP', 'ExtraTrees', 'HistGradientBoosting', 'Huber', 
          'Stacking', 'KernelRidge', 'ExtraTree', 
          'AdaBoost', 'Voting']

    
    generators = {model: Generator(train_df_rp_2, model_type=model) for model in models}

    directions = ['oeste', 'leste', 'sul']

    output_dir = './resultados_rp_alinhados_moda'
    os.makedirs(output_dir, exist_ok=True)

    

    for model_name, generator in generators.items():
        for direction in directions:
            print("opa")
            df_result = generator.gera_df(train_df_norte, direction=direction)
            file_name = f"{model_name}_{direction}.csv"
            file_path = os.path.join(output_dir, file_name)
            df_result.to_csv(file_path, index=False)
            print(f"DataFrame para o modelo {model_name} e direção {direction} salvo em {file_path}")
