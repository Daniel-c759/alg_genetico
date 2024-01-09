import numpy as np
from funciones_redes import red_completa
from sklearn.metrics import f1_score

def gen_a_diccionario(gen:list,var_redes:int,var_const:int)->dict:
    '''
    Función encargada de pasar un gen del algoritmo genetico a un diccionario para usar
    junto a las funciones de redes
    ------------------------------
    gen: lista en la cual se encuentran los distintos pesos y bias de las redes para la 
    predicción final
    var_redes: cantidad de variables que se estiman en la red LSTM
    var_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    ----------------------------------
    RETURN
    dict0: diccionario con pesos y bias de la red LSTM
    dict1: diccionario con pesos y bias de las variables involucradas en la estimación
    final de la red categorica, esto incluye las variables estimadas con LSTM
    '''
    dict0={}
    for i in range(1,var_redes+1):
        indice0=(i-1)*12
        indice1=i*12
        params={"w01":gen[indice0:indice1][0],
                "w11":gen[indice0:indice1][1],
                "b11":gen[indice0:indice1][2],
                "w02":gen[indice0:indice1][3],
                "w12":gen[indice0:indice1][4],
                "b12":gen[indice0:indice1][5],
                "w03":gen[indice0:indice1][6],
                "w13":gen[indice0:indice1][7],
                "b13":gen[indice0:indice1][8],
                "w04":gen[indice0:indice1][9],
                "w14":gen[indice0:indice1][10],
                "b14":gen[indice0:indice1][11]}
        dict0["var"+str(i)]=params
    if var_const==0:
        return dict0
    dict1={}
    for i in range(var_redes+var_const+1,0,-1):
        total=var_redes+var_const+2
        dict1["wvar"+str(total-i)]=gen[-i]
    return dict0, dict1

def crear_individuo(genetic_pool:'list | np.ndarray',
                    tamano_red:int,tamano_const:int)->list:
    '''
    Función encargada de crear un gen a partir de una lista de opción
    ---------------------------------
    genetic_pool: lista o array de la cual se sacaran valores que seran parte del gen de
    un individuo
    tamano_red: cantidad de variables que se estiman en la red LSTM
    tamano_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    -----------------------------
    RETURN
    individuo: regresa el gen de un individuo
    '''
    individuo=[]
    if tamano_const==0:
        individuo+=[np.random.choice(genetic_pool,12*tamano_red)]
        return individuo[0]
    else:
        individuo+=[np.random.choice(genetic_pool,
                                         12*tamano_red+tamano_red+tamano_const+1)]
        return individuo[0]

def crear_poblacion(genetic_pool:'list | np.ndarray',var_redes:int, var_const:int,
                    tamano_poblacion:int)->list:
    '''
    Función encargada de crear una población que sera evaluada en sus genes
    ------------------------------
    genetic_pool: lista o array de la cual se sacaran valores que seran parte del gen de
    un individuo
    var_redes: cantidad de variables que se estiman en la red LSTM
    var_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    tamano_poblacion: cantidad de individuos de la población
    -------------------------------------------
    RETURN
    poblacion:list Lista con los genes 
    '''
    poblacion=[]
    for i in range(tamano_poblacion):
        ind=crear_individuo(genetic_pool=genetic_pool,
                            tamano_red=var_redes,
                            tamano_const=var_const)
        poblacion.append(ind)
    return poblacion

def fitness_poblacion(poblacion:list, data_red:'list | np.ndarray',data_const:np.ndarray,
                      var_redes:int, var_const:int, verdadero:np.ndarray):
    '''
    Función encargada de medir el desempeño de una población candidata de solución
    ---------------------------------------------------------
    poblacion: lista con los genes a ser puestos a prueba
    data_red: array de numpy o list con la información necesaria de las variables y 
    observaciones de cada individuo para realizar la red LSTM
    data_const: array con información de un individuo y las variables asociadas a este para
    la predicción binaria
    var_redes: cantidad de variables que se estiman en la red LSTM
    var_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    verdadero: array con los valores verdaderos (categorias 1 ó 0)
    -------------------------------------------------------
    RETURN
    prob_reproduccion: array con las probabilidad de reproducción del gen
    max_valor:int valor maximo del fitness dentro de la población
    mejores_params:list lista con los mejores pesos y bias para los datos recolectados 
    '''
    fitness=[]
    for i in poblacion:
        dicc_red, dicc_const=gen_a_diccionario(gen=i,var_redes=var_redes,
                                               var_const=var_const)
        prediccion=red_completa(data_temp=data_red,data_const=data_const,
                                pesos_lstm=dicc_red,pesos_const=dicc_const)
        metrica=f1_score(y_true=verdadero,y_pred=prediccion,average='macro')
        fitness.append(metrica)
    fitness=np.array(fitness)
    poblacion=np.array(poblacion)
    max_valor=fitness[np.where(fitness==fitness.max())[0][0]]
    mejores_params=poblacion[np.where(fitness==fitness.max())[0][0]]
    prob_reproduccion=fitness/fitness.sum()
    return prob_reproduccion, max_valor, mejores_params

def reproduccion(poblacion:list,prob_reproduccion:np.ndarray,
                 var_redes:int, var_const:int)->list:
    '''
    Función encargada de generar la decendencia de una población de genes, esta cuenta con
    una probabilidad de reproducción para cada gen
    -------------------------------------------------------
    poblacion: lista con los genes a ser puestos a prueba
    prob_reproduccion: array en el cual se encuentra la probabilidad de que el gen se 
    multiplique en una siguiente generación
    var_redes: cantidad de variables que se estiman en la red LSTM
    var_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    verdadero: array con los valores verdaderos (categorias 1 ó 0)
    -------------------------------------------------------
    RETURN
    offspring: List, lista de arrays la decendencia de la población original insertada
    '''
    tamano_pob=len(poblacion)
    #desendencia
    offspring=[]
    for i in range(tamano_pob//2):
        padres=np.random.choice(tamano_pob,2,p=prob_reproduccion)
        cross_point=np.random.randint(12*var_redes+var_redes+var_const+1)
        offspring+=[np.append(poblacion[padres[0]][:cross_point],poblacion[padres[1]][cross_point:])]
        offspring+=[np.append(poblacion[padres[1]][:cross_point],poblacion[padres[0]][cross_point:])]
    return offspring

def mutar(poblacion:list, prob:float, pool:'list | np.ndarray'):
    '''
    Función encargada de generar mutaciones en la población basada en una posibilidad de 
    mutar
    ---------------------------------------------------------------
    poblacion: lista con los genes a ser puestos a prueba
    prob: probabilidad de que se realice una mutación
    pool:lista o array de la cual se sacaran valores que seran parte del gen de
    un individuo
    -----------------------------------------------------------------
    RETURN
    poblacion: List o array, de la población generada tras la mutación
    '''
    for i in range(len(poblacion)):
        ind_mutar=poblacion[i]
        if np.random.random()<prob:
            mutacion=np.random.choice(pool)
            ind_mutar=np.append([mutacion],ind_mutar[1:])
        for j in range(1,len(ind_mutar)-1):
            if np.random.random()<prob:
                mutacion=np.random.choice(pool)
                ind_mutar_pre=np.append(ind_mutar[0:j],[mutacion])
                ind_mutar=np.append(ind_mutar_pre,ind_mutar[j+1:])
        poblacion[i]=ind_mutar
    return poblacion

def optimizar_gen(genetic_pool:'list | np.ndarray',var_redes:int, var_const:int,
                    tamano_poblacion:int,data_red:'list | np.ndarray',
                    data_const:np.ndarray,verdadero:np.ndarray,prob:float,
                    generaciones:int, tol:float, max_intentos:int):
    '''
    Función para encontrar la mejor configuración de la red mediante iteraciones con 
    varias generaciones de las poblaciones
    --------------------------------------------------------------
    genetic_pool: lista o array de la cual se sacaran valores que seran parte del gen de
    los individuos de la poblacion
    var_redes: cantidad de variables que se estiman en la red LSTM
    var_const: cantidad de variables que se usan en la red categorica sin incluir las 
    estimadas en la red LSTM
    tamano_poblacion: cantidad de individuos de la población
    data_red: array de numpy o list con la información necesaria de las variables y 
    observaciones de cada individuo para realizar la red LSTM
    data_const: array con información de un individuo y las variables asociadas a este para
    la predicción binaria
    verdadero: array con los valores verdaderos (categorias 1 ó 0)
    prob: probabilidad de que se realice una mutación
    generaciones: entero que indica el número de generaciones que se quiere probar en el
    algoritmo genetico
    tol: indica cuanto debe mejorar el valor optimo para seguir probando iteraciones
    max_intentos: maximo de intentos sin mejorar antes de parar abructamente
    ---------------------------------------------------------------------------------
    RETURN
    valor_opt: int, valor máximo alcanzado por el gen
    param_opt: array con la configuración optima de genes
    '''
    poblacion=crear_poblacion(genetic_pool,var_redes,var_const,tamano_poblacion)
    valor_opt=0
    param_opt=0
    intentos=max_intentos
    for i in range(generaciones):
        print(f"Esta es la generación {i+1} de {generaciones}")
        if max_intentos==0:
            print("CRITERIO DE PARADA TEMPRANA ALCANZADO")
            break
        prob_reproduccion,max_valor, mejores_params=fitness_poblacion(poblacion,
                                                                  data_red,
                                                                  data_const,
                                                                  var_redes,var_const,
                                                                  verdadero)
        if max_valor-valor_opt<tol:
            max_intentos-=1
        else:
            max_intentos=intentos
            valor_opt=max_valor
            param_opt=mejores_params
        decendencia=reproduccion(poblacion,prob_reproduccion,var_redes,var_const)
        decendencia=mutar(decendencia,prob,genetic_pool)
        poblacion=decendencia
    return valor_opt,param_opt

# verdad=np.array([1,0,1,0,0,0,1])
# print(verdad[2:])

# variables=3

 
# poblacion=crear_poblacion(np.linspace(-2,2,50),variables,2,5)
# # print(poblacion[0].tolist())
# print(type(poblacion[:5]))


# # print(gen_a_diccionario(poblacion[0],variables,2))

# print(mutar(poblacion,0.0005,np.linspace(-2,2,50)))
