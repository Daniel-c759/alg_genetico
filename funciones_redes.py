from math import exp, tanh
import numpy as np

def suma_ponderada(valor0:float,
                    w0:float,
                    valor1:float,
                    w1:float, bias:float,
                    funcion:str)->float:
    '''
    Función encargada de realizar las sumas ponderadas correspondientes a la LSTM, 
    en este caso, la función final es exponencial "exp" o tangente hiperbolico "tanh"
    ---------------
    valor0: Primer valor que se multiplica con w0
    w0: Peso asociado al valor0
    valor1: Primer valor que se multiplica con w1
    w1: Peso asociado al valor1
    ---------------------
    RETURN:
    salida: Float, variable transformada por medio de una función sigmoidea exponencial o
    sigmoidea tangente hiperbolic
    '''
    entrada=valor0*w0+valor1*w1+bias
    if funcion=="exp":
        salida=1/(1+exp(-entrada))
        return salida
    elif funcion=="tanh":
        salida=tanh(entrada)
        return salida
    else:
        print("Elección incorrecta de la función")
        exit()

def un_paso_red(short_memory:float,
                long_memory:float,
                input1:float,
                w01:float,w11:float,b11:float,
                w02:float,w12:float,b12:float,
                w03:float,w13:float,b13:float,
                w04:float,w14:float,b14:float):
    '''
    Función encargada de realizar un paso de la red LSTM, en este caso se solicita la 
    memoria a corto plazo y largo plazo.
    -------------------
    short_memory: Memoria a corto plazo de la red LSTM
    long_memory: Memoria a largo plazo de la red LSTM
    input1: Valor de la variable en un punto de la red
    w01: Peso asociado a la memoria a corto plazo para el porcentaje que pasa a largo 
    plazo
    w11: Peso asociado al input1 que viene de la red
    b11: bias asociado a la red que compone las operaciones para el porcentaje que pasa 
    a largo plazo
    w02: Peso asociado al porcentaje a recordar de la memoria a corto plazo a largo plazo
    w12: Peso asociado al porcentaje a recordar del input1 a largo plazo
    b12: bias asociado a la red que compone las operaciones para el porcentaje a recordar
    a largo plazo
    w03: Peso asociado al potencial a recordar de la memoria a corto plazo a largo plazo
    w13: Peso asociado al potencial a recordar del input1 a largo plazo
    b13: bias asociado a la red que compone las operaciones para el potencial a recordar
    a largo plazo
    w04: Peso asociado al potencial a recordar de la memoria a corto plazo
    w14: Peso asociado al potencial a recordar a corto plazo del imput1
    b14 bias asociado a la red que compone las operaciones del potencial a recordar del 
    corto plazo
    ---------------------
    RETURN
    short_memory:float, memoria a corto plazo, al final del ciclo es la predicción a un 
    paso del nuevo valor
    long_memory: float, memoria a largo plazo
    '''
    perc_long_memory=suma_ponderada(short_memory,w01,input1,w11,b11,"exp")
    long_memory=long_memory*perc_long_memory
    perc_to_remember=suma_ponderada(short_memory,w02,input1,w12,b12,"exp")
    poten_to_remember=suma_ponderada(short_memory,w03,input1,w13,b13,"tanh")
    long_memory=long_memory+perc_to_remember*poten_to_remember
    poten_to_remember_sh=suma_ponderada(short_memory,w04,input1,w14,b14,"exp")
    short_memory=tanh(long_memory)*poten_to_remember_sh
    return short_memory,long_memory

def red_lstm(data:'list | np.ndarray', indice:int,
             w01:float,w11:float,b11:float,
                w02:float,w12:float,b12:float,
                w03:float,w13:float,b13:float,
                w04:float,w14:float,b14:float)->float:
    '''
    Función encargada de realizar para un individuo y una variable, la red LSTM, en este
    caso la memoria a corto y largo plazo se inicia en ceros
    ----------------------------------------
    data: array de numpy o list con los valores de las variables y los puntos del tiempo
    de un individuo
    indice: Necesario para indicar sobre que variable se realiza la red LSTM
    w01: Peso asociado a la memoria a corto plazo para el porcentaje que pasa a largo 
    plazo
    w11: Peso asociado al input1 que viene de la red
    b11: bias asociado a la red que compone las operaciones para el porcentaje que pasa 
    a largo plazo
    w02: Peso asociado al porcentaje a recordar de la memoria a corto plazo a largo plazo
    w12: Peso asociado al porcentaje a recordar del input1 a largo plazo
    b12: bias asociado a la red que compone las operaciones para el porcentaje a recordar
    a largo plazo
    w03: Peso asociado al potencial a recordar de la memoria a corto plazo a largo plazo
    w13: Peso asociado al potencial a recordar del input1 a largo plazo
    b13: bias asociado a la red que compone las operaciones para el potencial a recordar
    a largo plazo
    w04: Peso asociado al potencial a recordar de la memoria a corto plazo
    w14: Peso asociado al potencial a recordar a corto plazo del imput1
    b14 bias asociado a la red que compone las operaciones del potencial a recordar del 
    corto plazo
    ---------------------
    RETURN
    short_memory: Flota, predicción de la red para una variable
    '''
    short_memory=0
    long_memory=0
    for i in range(len(data)):
        short_memory,long_memory=un_paso_red(short_memory,long_memory,data[i][indice],
                    w01,w11,b11,
                    w02,w12,b12,
                    w03,w13,b13,
                    w04,w14,b14)
    return short_memory

def multi_red_lstm(data:'list | np.ndarray',pesos_vars:dict)->list:
    '''
    Función encargada de realizar para un individuo la predicción de una red LSTM para
    cada variable asociada al individuo
    --------------------------
    data: array de numpy o list con los valores de las variables y los puntos del tiempo
    de un individuo
    pesos_vars: diccionario con los pesos y bias de cada una de las variables
    ----------------------------
    RETURN
    resultados: list, lista con las predicciones a un paso de cada variable
    '''
    variables=data.shape[1]
    indice=0
    resultados=[]
    for value in pesos_vars.values():
        if indice>=variables:
            print("ERROR EN DIMENSIONES DE VARIABLES E ITEMS DEL DICCIONARIO pesos_vars")
            exit()
        pred=red_lstm(data=data,indice=indice,**value)
        resultados.append(pred)
        indice+=1
    return resultados

def red_categorica(data:np.ndarray,**kwargs)->int:
    '''
    Función encargada de realizar la predicción binaria para unas variables
    --------------------------
    data: array con información de un individuo y las variables asociadas a este
    **kwargs: diccionario con pesos asociados a cada una de las variables
    --------------------------
    RETURN
    Regresa 1 o 0 dependiendo el valor de la función signoideal exponencial
    '''
    parcial=[]
    contador=1
    for i,j in zip(data,kwargs.values()):
        if contador>=len(kwargs):
            parcial.append(1*j)
        multiplicacion=i*j
        parcial.append(multiplicacion)
        contador+=1
    parcial=np.array(parcial)
    salida=1/(1+exp(-parcial.sum()))
    if salida>0.5:
        return 1
    else:
        return 0
    
def red_completa_ind(data_lstm:'list | np.ndarray',
                     data_const:np.ndarray,
                     pesos_lstm:dict,
                     pesos_const:dict)->int:
    '''
    Función encargada de realizar las predicciones de las variables de la red LSTM y
    utilizarlas en la red categorica para predecir una etiqueta de 1 o 0 segun corresponda
    --------------------------------------------------
    data_lstm: array de numpy o list con los valores de las variables y los puntos del 
    tiempo de un individuo para la red LSTM
    data_const: array con información de un individuo y las variables asociadas a este para
    la predicción binaria
    pesos_lstm: diccionario con los pesos y bias de cada una de las variables asociadas a
    la red LSTM
    pesos_const: diccionario con pesos asociados a cada una de las variables para la 
    predicción binaria
    ---------------------------------------
    RETURN
    resultado: Predicción final sobre si el individuo pertenece a 1 o 0
    '''
    pred=multi_red_lstm(data=data_lstm,pesos_vars=pesos_lstm)
    data=np.append(pred,data_const)
    resultado=red_categorica(data=data,**pesos_const)
    return resultado

def red_completa(data_temp:'list | np.ndarray',
                     data_const:np.ndarray,
                     pesos_lstm:dict,
                     pesos_const:dict)->np.ndarray:
    '''
    Función encargada de tomar varias observaciones de varios individuos y hacer 
    predicción binaria
    --------------------------------------------------------
    data_temp: array de numpy o list con la información necesaria de las variables y 
    observaciones de cada individuo para realizar la red LSTM
    data_const: array con información de un individuo y las variables asociadas a este para
    la predicción binaria
    pesos_lstm: diccionario con los pesos y bias de cada una de las variables asociadas a
    la red LSTM
    pesos_const: diccionario con pesos asociados a cada una de las variables para la 
    predicción binaria
    ---------------------------------------
    RETURN
    resultado: Array con las predicciones finales de las observaciones, que permite saber
    si el individuo pertenece a 1 o 0
    '''
    resultados=[]
    for i in data_temp:
        resultados_ind=red_completa_ind(data_lstm=i, data_const=data_const,
                                        pesos_lstm=pesos_lstm, pesos_const=pesos_const)
        resultados.append(resultados_ind)
    return np.asarray(resultados)