import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sqlite3
import category_encoders as ce
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.model_selection import train_test_split #Import train_test_split function
import matplotlib.pyplot as plt #Representación de gráficos
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                            ConfusionMatrixDisplay) #Métricas
from imblearn.over_sampling import SMOTE

def dataset():
    # Conectar a la base de datos SQLite
    conn = sqlite3.connect('DataBase/Alumn_Data.db')

    # Consulta SQL para seleccionar todos los datos de la tabla Alumn_Data
    query = "SELECT * FROM Alumn_Data"

    # leer la tabla y convertirla en un DataFrame
    df = pd.read_sql_query(query, conn)

    # Cerrar la conexión a la base de datos
    conn.close()

    #Mapeamos la salida de promociona - Clase objetivo.
    # Define un diccionario para mapear 'N' a 0 y 'S' a 1
    mapeo = {'N': 0, 'S': 1}

    # Aplica el mapeo al campo deseado
    df['PROMOCIONA'] = df['PROMOCIONA'].map(mapeo)    
    
    #Realización de OneHotEncoder
    encoder=ce.OneHotEncoder(cols=['A_PAIS', 'P_PAIS', 'M_PAIS', 'P_EDAD', 'M_EDAD', 'P_TRABAJO', 'M_TRABAJO', 'A_RESIDENCIA'],handle_unknown='return_nan',return_df=True,use_cat_names=True)
    df_encoded = encoder.fit_transform(df)

    # Convertimos las columnas resultantes de OneHotEncoder de float a int
    encoded_columns = df_encoded.columns.difference(df.columns)
    df_encoded[encoded_columns] = df_encoded[encoded_columns].astype('int64')
    
    #Eliminamos los campos ID y CREATED que son auto y sin información para modelo
    df_encoded.drop(columns=['ID','CREATED'], inplace=True)

    #Convertimos a enteros los campos que no son media
    additional_columns = [
        'ESO1_C_1EV', 'ESO1_C_2EV', 'ESO1_C_EVF',
        'ESO1_EF_1EV', 'ESO1_EF_2EV', 'ESO1_EF_EVF',
        'ESO1_I_1EV', 'ESO1_I_2EV', 'ESO1_I_EVF',
        'ESO1_M_1EV', 'ESO1_M_2EV', 'ESO1_M_EVF',
        'ESO2_C_1EV', 'ESO2_C_2EV', 'ESO2_C_EVF',
        'ESO2_EF_1EV', 'ESO2_EF_2EV', 'ESO2_EF_EVF',
        'ESO2_I_1EV', 'ESO2_I_2EV', 'ESO2_I_EVF',
        'ESO2_M_1EV', 'ESO2_M_2EV', 'ESO2_M_EVF',
        'FPB1_P_1EV', 'FPB1_P_2EV', 'FPB1_P_EVF',
        'FPB1_CA_1EV', 'FPB1_CA_2EV', 'FPB1_CA_EVF',
        'FPB1_CS_1EV', 'FPB1_CS_2EV', 'FPB1_CS_EVF',
        'REP_1ESO', 'REP_2ESO', 'REP_1FPB',
        'REC_1EV', 'REC_2EV', 'REC_EVEX'
    ]
 
    # Combina encoded_columns y additional_columns en una sola lista
    columns_to_convert = list(encoded_columns) + additional_columns

    # Convierte las columnas a int64
    df_encoded[columns_to_convert] = df_encoded[columns_to_convert].astype('int64')

    # Imprime para verificar la conversión
    print(df_encoded[columns_to_convert].dtypes) 
    
    return df_encoded, encoded_columns

def outliers(df):
    print("Entra outliers")
    # Crear un DataFrame para almacenar los outliers
    outliers = pd.DataFrame()

    # Diccionario para almacenar columnas con outliers
    columnas_con_outliers = {}

    # Identificar las columnas que no son resultado de One Hot Encoding
    columnas_no_one_hot = [col for col in df.columns if not set(df[col].dropna().unique()).issubset({0, 1})]

    # Calcular outliers para cada columna numérica que no sea One Hot Encoded
    for column in df[columnas_no_one_hot].select_dtypes(include=['float64', 'int64']).columns:
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
        
        # Identificar los outliers utilizando los límites de los percentiles
        outliers[column] = ((df[column] < lower_bound) | (df[column] > upper_bound))
        
        # Añadir la columna al diccionario si tiene outliers
        if outliers[column].any():
            columnas_con_outliers[column] = df[column][outliers[column]]

    # Mostrar el resumen de las columnas con outliers
    # for col, outliers in columnas_con_outliers.items():
    #     print(f"Columna '{col}' tiene {len(outliers)} outliers.")

    # Mostrar las columnas con outliers
    # print("\nColumnas que tienen outliers:")
    # print(list(columnas_con_outliers.keys()))    
    
    # DataFrame para almacenar los valores winsorizados
    df_winsorized = df.copy()

    # Aplicar la winsorización usando percentiles
    for column in columnas_con_outliers:
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
        
        # Imprimir los límites para la verificación
        print(f"Column: {column}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        
        # Winsorizar los valores fuera de los límites usando percentiles
        df_winsorized[column] = winsorize(df[column], limits=[0.01, 0.01]) #1% por arriba y por abajo    
    
    return df_winsorized

def norm_estand(data):
    print('Entra normalización')

    #Estandarizar
    columns_to_stan_norm = [
        'ESO1_MEDIA_1EV', 'ESO1_MEDIA_2EV', 'ESO1_MEDIA_EVF', 
        'ESO2_MEDIA_1EV', 'ESO2_MEDIA_2EV', 'ESO2_MEDIA_EVF', 
        'FPB1_MEDIA_1EV', 'FPB1_MEDIA_2EV', 'FPB1_MEDIA_EVF'
    ]

    # Comprobar si las columnas existen en el DataFrame
    missing_columns = [col for col in columns_to_stan_norm if col not in data.columns]

    if missing_columns:
        print(f"Las siguientes columnas no se encuentran en el DataFrame: {missing_columns}")
    else:
        # Crear una instancia del StandardScaler
        scaler = StandardScaler()

        # Aplicar la estandarización solo a las columnas especificadas
        data[columns_to_stan_norm] = scaler.fit_transform(data[columns_to_stan_norm])

        # Verificar el resultado
        #print(data[columns_to_stan_norm])    
    
    # Normalizar

    # Comprobar si las columnas existen en el DataFrame
    missing_columns = [col for col in columns_to_stan_norm if col not in data.columns]

    if missing_columns:
        print(f"Las siguientes columnas no se encuentran en el DataFrame: {missing_columns}")
    else:
        # Crear una instancia del MinMaxScaler
        scaler = MinMaxScaler()

        # Aplicar la normalización solo a las columnas especificadas
        data[columns_to_stan_norm] = scaler.fit_transform(data[columns_to_stan_norm])
        
        # Verificar el resultado
        #print(data[columns_to_stan_norm].head())
    #print(data.info())
    
    return data

def select_v(data):
    print("Entra select")
    # SEPARO TODOS LOS DATOS DE FPB QUE NO APORTA INFO.

    # Lista de nombres de las columnas a separar
    columnas_separar = ['FPB1_MEDIA_1EV','FPB1_MEDIA_2EV','FPB1_P_1EV','FPB1_P_2EV','FPB1_CA_1EV',
                        'FPB1_CA_2EV','FPB1_CS_1EV','FPB1_CS_2EV', 'FPB1_MEDIA_EVF', 'FPB1_P_EVF',
                        'FPB1_CA_EVF', 'FPB1_CS_EVF']

    # Crear un nuevo DataFrame con solo las columnas seleccionadas
    df_sinFPB = data.copy()

    df_sinFPB.drop(columns=columnas_separar, inplace=True)    
    
    print("Empieza p-value")
    #SELECCIONO UN VALOR DE K MUESTRAS PARA UN P-VALUE CONCRETO: 0.2
    np.seterr(divide='warn', invalid='warn')

    #Divido el conjunto de datos en 80 entrenamiento y 20 test

    X_train, X_test, y_train, y_test = train_test_split(df_sinFPB.drop(labels=['PROMOCIONA'], axis=1),df_sinFPB['PROMOCIONA'], test_size=0.2, random_state=0)

    names=pd.DataFrame(X_train.columns)
    max_i = X_train.shape[1]

    p_val=0.2

    print("Empieza selectkbest")
    model = SelectKBest(score_func= f_classif)
    results = model.fit(X_train, y_train)
    results_df=pd.DataFrame(results.pvalues_)
    scored=pd.concat([names,results_df], axis=1)
    scored.columns = ['Feature', 'P_Values']
    scored=scored.sort_values(by=['P_Values'])

    for i in range(0, max_i):

    # Seleccionamos el p-valor en la iteración específica de k
        p_valor_iteracion_concreta = scored.iloc[i]['P_Values']

        if (p_valor_iteracion_concreta>p_val): #P-VALUE <=0.2
            break
            
        print("El p-valor en la iteración", i, "es:", p_valor_iteracion_concreta)
        
    k_selected=i
    print("k_selected:", k_selected)

    #19 características con p-value no superior a 0.2    

    #Ejecuto para k=19

    names=pd.DataFrame(X_train.columns)

    model = SelectKBest(score_func= f_classif, k=k_selected)
    results = model.fit(X_train, y_train)

    #creo un nuevo dataset solo con las columnas seleccionadas y la variable objetivo PROMOCIONA.

    # Obtiene las columnas seleccionadas
    selected_columns = X_train.columns[results.get_support()]

    # Conserva solo las columnas seleccionadas y la variable objetivo en el DataFrame original
    df_sinFPB_sel = df_sinFPB[selected_columns.append(pd.Index(['PROMOCIONA']))]
    
    #print(df_sinFPB_sel.info())
    
    return df_sinFPB_sel, selected_columns

def entrenamientoDT(data):
    print("Entra entrenamiento")
    
    #Generamos datos y equilibramos clases
    # Separar las características (X) y la variable objetivo (y)
    X = data.drop(columns='PROMOCIONA')
    y = data['PROMOCIONA']

    # Aplicar SMOTE para aumentar las muestras equiparando clases - balanceo
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convertir a DataFrame
    data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_resampled['PROMOCIONA'] = y_resampled

    # Verificar y listar las columnas
    print("Columnas del DataFrame:")
    print(data.columns)

    # Identificar las columnas binarias, ordinales y otras numéricas
    binary_columns = [col for col in X.columns if set(data[col].unique()).issubset({0, 1})]
    decimal_columns = ['ESO1_MEDIA_1EV', 'ESO1_MEDIA_EVF']  # Columnas numéricas con decimales
    ordinal_columns = ['A_IDIOMA_NVL', 'P_ESTUDIOS']  # Columnas ordinales
    integer_columns = [col for col in X.columns if col not in binary_columns + decimal_columns + ordinal_columns]

    # Crear datos artificiales mediante réplicas con perturbaciones adecuadas
    def create_perturbations(df, binary_cols, decimal_cols, ordinal_cols, integer_cols, n_copies=5, noise_level=0.01):
        data_augmented = df.copy()
        for i in range(n_copies):
            noisy_data = df.copy()
            for column in decimal_cols:
                if df[column].dtype in [np.float64, np.int64]:  # Aplicar ruido solo a columnas numéricas con decimales
                    noise = np.random.normal(0, noise_level, size=df[column].shape)
                    noisy_data[column] += noise
            for column in integer_cols:
                if df[column].dtype in [np.float64, np.int64]:  # Aplicar ruido a columnas de enteros y redondear
                    noise = np.random.normal(0, noise_level, size=df[column].shape)
                    noisy_data[column] = np.round(noisy_data[column] + noise).astype(int)
            for column in ordinal_cols:
                if column == 'A_IDIOMA_NVL':  # Asegurar límites de 1, 2, 3
                    noise = np.random.normal(0, noise_level, size=df[column].shape)
                    noisy_data[column] = np.clip(np.round(noisy_data[column] + noise).astype(int), 1, 3)
                elif column == 'P_ESTUDIOS':  # Asegurar límites de 1 a 7
                    noise = np.random.normal(0, noise_level, size=df[column].shape)
                    noisy_data[column] = np.clip(np.round(noisy_data[column] + noise).astype(int), 1, 7)
            data_augmented = pd.concat([data_augmented, noisy_data], axis=0)
        return data_augmented

    data_augmented = create_perturbations(data_resampled, binary_columns, decimal_columns, ordinal_columns, integer_columns, n_copies=30, noise_level=0.01)

    # Asegurarse de que las columnas binarias sigan siendo binarias
    for column in binary_columns:
        data_augmented[column] = data_augmented[column].round().clip(0, 1)

    #1178 observaciones
    #print(data_augmented.info())
    
    #Dividimos el conjunto en train y test
    X_train, X_test, y_train, y_test = train_test_split(data_augmented.drop(labels=['PROMOCIONA', ], axis=1),data_augmented['PROMOCIONA'], test_size=0.2, random_state=1)
    #Mostramos los subconjuntos
    print('Datos de entrenamiento (Atributos) -> X_train shape is: \n' , X_train.shape)
    print('Datos de test (Atributos) -> X_test shape is: \n' , X_test.shape)
    print('Datos de entrenamiento (Clase) -> y_train shape is: \n' , y_train.shape)
    print('Datos de test (Clase) -> y_test shape is: \n' , y_test.shape)
    print('Balanceo de la clase para entrenamiento:')
    print(y_train.value_counts())
    print('Balanceo de la clase para entrenamiento:')
    print(y_test.value_counts())

    # Modelo ajustado
    best_dt_model = DecisionTreeClassifier(
        ccp_alpha=0.0,
        class_weight=None,  # Las clases están balanceadas
        criterion='gini',
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=4,
        min_samples_split=15,
        min_weight_fraction_leaf=0.0,
        random_state=1,
        splitter='best'
    )

    #Entrenamiento
    best_dt_model.fit(X_train,y_train)
    
    #Evaluacion
    y_pred_dt = best_dt_model.predict(X_test)    
    
    #Evaluamos ACC sobre los datos de entrenamiento y los de test y comparamos
    training_prediction = best_dt_model.predict(X_train)

    validation_prediction = best_dt_model.predict(X_test)

    print('Exactitud training data: ', accuracy_score(y_true=y_train, y_pred=training_prediction))

    print('Exactitud validation data: ', accuracy_score(y_true=y_test, y_pred=validation_prediction))    

    #Función que devuelve las métricas de los modelos entrenados a partir de un y_test e y_pred
    def mx_display(y_test, y_pred, name_model, colour1, colour2):
        """
        Funcion que pinta la matriz de confusión y la normalizada
        
        Argumentos: mx y model_name
        Salida: NA
        """   
        #Matriz de confusión representada de forma visual-Modelo DecisionTree
        print('-----------------MATRICES DE CONFUSIÓN PARA MODELO %s:\n-----------------' % (name_model))
        mx = confusion_matrix(y_test, y_pred)
        
        #Añado etiqueta para que sea más visual la clase:
        mx_disp = ConfusionMatrixDisplay(confusion_matrix=mx, display_labels=['NO', 'SI'])

        #Añado etiqueta para que sea más visual la clase:
        mx_disp = ConfusionMatrixDisplay(confusion_matrix=mx, display_labels=['NO', 'SI'])
        #Cambiamos color
        fig, ax = plt.subplots()
        mx_disp.plot(cmap = colour1, ax=ax)
        # Guardar la figura
        img_path1 = f'static/{name_model}_confusion_matrix.png'
        plt.savefig(img_path1)
        plt.close(fig)

        #Matriz de confusión normalizada
        mx_disp_norm = confusion_matrix(y_test, y_pred, labels=[0, 1],  normalize='true')
        mx_disp_norm_disp = ConfusionMatrixDisplay(confusion_matrix=mx_disp_norm, display_labels=['NO', 'SI'])
        fig, ax = plt.subplots()
        mx_disp_norm_disp.plot(cmap = colour2, ax=ax)
        # Guardar la figura
        img_path2 = f'static/{name_model}_confusion_matrix_norm.png'
        plt.savefig(img_path2)
        plt.close(fig)
      

    #Evaluamos métricas de DT
    mx_display(y_test, y_pred_dt, "Decision Tree", 'Greens', 'Blues')

    return best_dt_model

def train_model():
    print("Entra train")
    df_encoded, encoded_columns = dataset()
    df_winsorized=outliers(df_encoded)
    df_norm=norm_estand(df_winsorized)
    df_sinFPB_sel,columnas_seleccionadas=select_v(df_norm)
    print(columnas_seleccionadas)
    best_dt_model=entrenamientoDT(df_sinFPB_sel)

    return(columnas_seleccionadas, encoded_columns, best_dt_model)

def run_prediction(csv_path):
    # Cargar el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(csv_path)
    
    #Realización de OneHotEncoder
    encoder=ce.OneHotEncoder(cols=['A_PAIS', 'P_PAIS', 'M_PAIS', 'P_EDAD', 'M_EDAD', 'P_TRABAJO', 'M_TRABAJO', 'A_RESIDENCIA'],handle_unknown='return_nan',return_df=True,use_cat_names=True)
    df_encoded = encoder.fit_transform(df)
    
    # Convertimos las columnas resultantes de OneHotEncoder de float a int
    encoded_columns = df_encoded.columns.difference(df.columns)

    df_encoded[encoded_columns] = df_encoded[encoded_columns].astype('int64')

    #Convertimos a enteros los campos que no son media
    additional_columns = [
        'ESO1_C_1EV', 'ESO1_C_2EV', 'ESO1_C_EVF',
        'ESO1_EF_1EV', 'ESO1_EF_2EV', 'ESO1_EF_EVF',
        'ESO1_I_1EV', 'ESO1_I_2EV', 'ESO1_I_EVF',
        'ESO1_M_1EV', 'ESO1_M_2EV', 'ESO1_M_EVF',
        'ESO2_C_1EV', 'ESO2_C_2EV', 'ESO2_C_EVF',
        'ESO2_EF_1EV', 'ESO2_EF_2EV', 'ESO2_EF_EVF',
        'ESO2_I_1EV', 'ESO2_I_2EV', 'ESO2_I_EVF',
        'ESO2_M_1EV', 'ESO2_M_2EV', 'ESO2_M_EVF',
        'FPB1_P_1EV', 'FPB1_P_2EV', 'FPB1_P_EVF',
        'FPB1_CA_1EV', 'FPB1_CA_2EV', 'FPB1_CA_EVF',
        'FPB1_CS_1EV', 'FPB1_CS_2EV', 'FPB1_CS_EVF',
        'REP_1ESO', 'REP_2ESO', 'REP_1FPB',
        'REC_1EV', 'REC_2EV', 'REC_EVEX'
    ]
 
    # Combina encoded_columns y additional_columns en una sola lista
    columns_to_convert = list(encoded_columns) + additional_columns

    # Convierte las columnas a int64
    df_encoded[columns_to_convert] = df_encoded[columns_to_convert].astype('int64') 
    
    #Añado los campos que no se generan con onehot y si en train, para compatibilidad
    
    #Estandarizar
    columns_to_stan_norm = [
        'ESO1_MEDIA_1EV', 'ESO1_MEDIA_2EV', 'ESO1_MEDIA_EVF', 
        'ESO2_MEDIA_1EV', 'ESO2_MEDIA_2EV', 'ESO2_MEDIA_EVF', 
        'FPB1_MEDIA_1EV', 'FPB1_MEDIA_2EV', 'FPB1_MEDIA_EVF'
    ]

    # Comprobar si las columnas existen en el DataFrame
    missing_columns = [col for col in columns_to_stan_norm if col not in df_encoded.columns]

    if missing_columns:
        print(f"Las siguientes columnas no se encuentran en el DataFrame: {missing_columns}")
    else:
        # Crear una instancia del StandardScaler
        scaler = StandardScaler()

        # Aplicar la estandarización solo a las columnas especificadas
        df_encoded[columns_to_stan_norm] = scaler.fit_transform(df_encoded[columns_to_stan_norm])

        # Verificar el resultado
        #print(data[columns_to_stan_norm])    
    
    # Normalizar

    # Comprobar si las columnas existen en el DataFrame
    missing_columns = [col for col in columns_to_stan_norm if col not in df_encoded.columns]

    if missing_columns:
        print(f"Las siguientes columnas no se encuentran en el DataFrame: {missing_columns}")
    else:
        # Crear una instancia del MinMaxScaler
        scaler = MinMaxScaler()

        # Aplicar la normalización solo a las columnas especificadas
        df_encoded[columns_to_stan_norm] = scaler.fit_transform(df_encoded[columns_to_stan_norm])
        
        # Verificar el resultado
        #print(data[columns_to_stan_norm].head())
    #print(df_encoded.info())

    # Lista de nombres de las columnas a separar
    columnas_separar = ['FPB1_MEDIA_1EV','FPB1_MEDIA_2EV','FPB1_P_1EV','FPB1_P_2EV','FPB1_CA_1EV',
                        'FPB1_CA_2EV','FPB1_CS_1EV','FPB1_CS_2EV', 'FPB1_MEDIA_EVF', 'FPB1_P_EVF',
                        'FPB1_CA_EVF', 'FPB1_CS_EVF']

    # Crear un nuevo DataFrame con solo las columnas seleccionadas
    df_sinFPB = df_encoded.copy()

    df_sinFPB.drop(columns=columnas_separar, inplace=True)  
    
    #obtenemos las columnas que usaremos para la predicción según el entrenamiento
    columnas_seleccionadas, train_encoded_columns, best_dt_model=train_model()
    
    missing_columns = list(set(train_encoded_columns) - set(encoded_columns))

    # Añadir las columnas faltantes en df_test con valor 0
    for col in missing_columns:
        df_sinFPB[col] = 0

    ids=df_sinFPB[['ID']].copy()
    
    # Conserva solo las columnas seleccionadas y la variable objetivo en el DataFrame original
    df_sinFPB_sel = df_sinFPB[columnas_seleccionadas]
    

    print("Hago la predicción")
    predictions = best_dt_model.predict(df_sinFPB_sel)

    # # Devolver los resultados asociados al ID (aquí podrías devolver la precisión, las predicciones, etc.)
    df_sinFPB_sel['ID']=ids
    
    # Crear un DataFrame con id y predicciones
    predictions_df = pd.DataFrame({
        'ID': df_sinFPB_sel['ID'],  # Suponiendo que tienes una columna 'id' en df_sinFPB_sel
        'RESULTADO': predictions  # Agregar las predicciones
    })

    # Reemplazar 1 y 0 con 'SI' y 'NO' respectivamente
    predictions_df['RESULTADO'] = predictions_df['RESULTADO'].map({1: 'SI PROMOCIONA', 0: 'NO PROMOCIONA'})

    # Mostrar el DataFrame con id y predicciones asociadas
    print(predictions_df)
    return(predictions_df)