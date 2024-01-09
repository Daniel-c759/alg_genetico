import pandas as pd
import numpy as np



def agregar_info_redes(data:pd.DataFrame,excluir:list,identificador:str)->list:
    '''
    Función generada para pasar la información de un DataFrame de pandas a una
    estructura que pueda usarse en una red neuronal tipo lstm, solo se reciben 
    variables numericas.
    En esta función se necesita tener un identificador de individuo, con el cual
    se va a obtener la matriz de variables de éste. Cada individuo puede tener
    diferente número de observaciones pero debe contar con el mismo número de 
    variables
    '''
    # lista donde se almacena la información
    lista_data=[]
    # ciclo basado en los identificadores para obtener las observaciones  y vars
    for i in data[identificador].value_counts().index:
        # Se filtran las observaciones del identificador i
        dataFil=data.loc[data[identificador]==i,:]
        # Se convierte el DataFrame excluyendo variables no necesarias
        arreglo=np.asarray(dataFil.drop(columns=excluir))
        # Agregar el array a la lista_data
        lista_data.append(arreglo)
    return lista_data



def array_redes(data:pd.DataFrame,excluir:list, identificador:str)->np.ndarray:
    '''
    Función generada para pasar la información de un DataFrame de pandas a una
    estructura que pueda usarse en una red neuronal tipo lstm, solo se reciben 
    variables numericas.
    En esta función se necesita tener un identificador de individuo, con el cual
    se va a obtener la matriz de variables de éste. Cada individuo debe tener
    el mismo número de observaciones y contar con el mismo número de variables.
    '''
    # Convertir a array para mejor manejo
    datanumpy=np.asarray(data.drop(columns=excluir))
    # Cuantos individuos tengo
    individuos=len(np.unique(data[identificador]))
    # Estructura de array 3d
    array_data=datanumpy.reshape((individuos,
                                  int(data.shape[0]/individuos),
                                  data.drop(columns=excluir).shape[1]
                                  ))
    return array_data



