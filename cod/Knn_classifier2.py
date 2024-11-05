from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss
)
import numpy as np
import pandas as pd
import os

def evaluate_knn_classifier(train_df_x, train_df_y, test_df_x, test_df_y, n_neighbors=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_df_x, train_df_y)

    y_pred_class = knn_classifier.predict(test_df_x)
    y_pred_proba = knn_classifier.predict_proba(test_df_x)

    accuracy = accuracy_score(test_df_y, y_pred_class)
    log_loss_value = log_loss(test_df_y, y_pred_proba)

    return {
        'Acurácia': accuracy,
        'Log Loss': log_loss_value,
    }


def load_model_solutions(diretorio, modelo, train_norte):
    print(f"\n Solução {modelo} \n")
    
    # Carregar os dados de cada direção
    df_leste = pd.read_csv(f"{diretorio}/{modelo}_leste.csv")
    df_sul = pd.read_csv(f"{diretorio}/{modelo}_sul.csv")
    df_oeste = pd.read_csv(f"{diretorio}/{modelo}_oeste.csv")
    
    # Concatenar os DataFrames
    df_solution = pd.concat([train_norte, df_oeste, df_sul, df_leste], ignore_index=True)

    x_solution = df_solution[['B1_min', 'B2_min', 'B3_min', 'B4_min', 'B5_min', 'B6_min', 'B7_min']]
    y_solution = df_solution['RP']
    
    return x_solution, y_solution

def main():
    train_df = pd.read_csv('resultados-all-beacons/train_df.csv')
    test_df = pd.read_csv('resultados-all-beacons/test_df.csv')

    metrics_list = []

    # Melhor Caso
    train_df_x_bestcase = train_df[['B1_min', 'B2_min', 'B3_min', 'B4_min', 'B5_min', 'B6_min', 'B7_min']]
    train_df_y_bestcase = train_df['RP']
    test_df_x_bestcase = test_df[['B1_min', 'B2_min', 'B3_min', 'B4_min', 'B5_min', 'B6_min', 'B7_min']]
    test_df_y_bestcase = test_df['RP']

    metrics = evaluate_knn_classifier(train_df_x_bestcase, train_df_y_bestcase, test_df_x_bestcase, test_df_y_bestcase)
    metrics['Modelo'] = "KNN - Melhor Caso"
    metrics_list.append(metrics)

    # Pior Caso
    train_norte = train_df[train_df['Direction'] == 'norte']
    train_df_x_worstcase = train_norte[['B1_min', 'B2_min', 'B3_min', 'B4_min', 'B5_min', 'B6_min', 'B7_min']]
    train_df_y_worstcase = train_norte['RP']
    
    metrics = evaluate_knn_classifier(train_df_x_worstcase, train_df_y_worstcase, test_df_x_bestcase, test_df_y_bestcase)
    metrics['Modelo'] = "KNN - Pior Caso"
    metrics_list.append(metrics)

    # Soluções dos outros modelos
    '''models = [
        'LinearRegression', 'KNN', 'DecisionTree', 'SVR', 
        'RandomForest', 'Ridge', 'Lasso', 'ElasticNet', 'GaussianProcess',
        'MLP', 'ExtraTrees', 'HistGradientBoosting', 'Huber', 
        'Stacking', 'KernelRidge', 'ExtraTree', 
        'AdaBoost', 'Voting', 'Polynomial'
    ] ''' 
   

    models = ['LinearRegression', 'KNN', 'DecisionTree', 'SVR', 
          'RandomForest', 'Ridge', 'Lasso', 'ElasticNet', 
          'MLP', 'ExtraTrees', 'HistGradientBoosting', 'Huber', 
          'Stacking', 'KernelRidge', 'ExtraTree', 
          'AdaBoost', 'Voting']

    
    diretorio = "resultados_rp_alinhados_moda"  

    for model_name in models:
        train_df_x_solution, train_df_y_solution = load_model_solutions(diretorio, model_name, train_norte)
        if not train_df_x_solution.empty:
            metrics = evaluate_knn_classifier(train_df_x_solution, train_df_y_solution, test_df_x_bestcase, test_df_y_bestcase)
            metrics['Modelo'] = f"{model_name} - Solução"
            metrics_list.append(metrics)


    # Salvando em CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values(by='Acurácia', ascending=False)

    metrics_df.to_csv('resultados_com_RP_alinhados_moda.csv', index=False)

if __name__ == '__main__':
    main()


