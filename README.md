# Data Science and Machine Learning Postgraduate Course: Capstone Project
Con el presente proyecto se pretende realizar un sistema de recomendación de productos a clientes basándose en la base de datos de H&M. Dicha base de datos consta de variables tales como id de cliente, id de artículo, fecha de compra, precio y canal de venta.
Utilizamos deep learning por la flexibilidad que tiene a la hora de realizar la ingeniería de características. El modelo se basa en un sistema de redes neuronales a fin de poder extraer las características más representativas de los datos para obtener las recomendaciones finales a los usuarios. El método utilizado es llamado factorización de matrices (en adelante, MF). Al igual que los modelos de Principal Component Analysis (PCA), el algoritmo MF utiliza una descomposición en dos matrices (una de customers y otra de articles), llamadas embeddings. Existen factores desconocidos que determinan el comportamiento de los usuarios que son capturados por medio de este método. Es necesario definir el número de factores latentes k que definen el resultado de la factorización.
Primeramente se realiza un breve análisis de la base de datos, a fin de observar si existen registros repetidos, artículos poco vendidos, etc.
Se ha determinado que se eliminarían los registros repetidos (ya que coinciden en usuario, artículo y fecha de compra, por lo que se entiende que se ha duplicado el registro por un fallo). Además, una vez realizado el análisis, se llega a la conclusión que es recomendable descartar los artículos que se han vendido menos de 10 veces (ya que esto implica que dichos productos no son deseados).
Posterior a esto, se separa la base de datos en dos conjuntos: train-validation set y test set. A diferencia de otros proyectos, aquí se propone separar el último producto comprado por cada usuario (utilizando la variable de tiempo) para el test set y, una vez realizado el modelado, calcular la cantidad de veces que el artículo separado se encuentra en el top 10 de productos recomendados al cliente. 
Inicialmente, se ha tratado de utilizar negative sampling para realizar un muestreo de productos que los usuarios podrían haber visto y decidido no comprar (considerando 4 “no compras” por cada compra realizada), sin embargo no se ha conseguido generar un modelo adecuado con esto. 
Para el modelado, se ha utilizado Keras de Tensor Flow, en particular tf.keras.Sequential. Para los layers se han realizado distintas pruebas y finalmente se han escogido StringLookup para el preprocesamiento, un layer Dense con función de activación tanh, un layer LSTM (Long Short-Term Memory) y finalmente un layer de Embedding. 
Por otra parte, es importante tener en cuenta el learning rate, el cual es un hiperparámetro que indica que tan largo será el camino que tomará el algoritmo de optimización. Si el valor es muy pequeño, se puede quedar estancado en un mínimo local; mientras que si el valor es muy alto el algoritmo se puede pasar de largo y no encontrar nunca el mínimo global.
Se ha comprobado que con la utilización de GPU local se ha obtenido una mejora en cuanto al tiempo de ejecución.
Dentro de las pruebas realizadas, se ha hecho lo siguiente:
*	Layers:
    * StringLookup y Embedding
    * StringLookup, Embedding y Dense (con función de activación Relu y tanh)
    * StringLookup, Embedding, Dense (con función de activación Relu y tanh) y LSTM
*	Algoritmo de optimización:
    * Adagrad
    * Adam
    * Adamax
*	Cantidad mínima de venta de los artículos:
    * Sin eliminar ningún artículo.
    * 10 ventas mínimas.
    * 20 ventas mínimas.
*	Tasa de aprendizaje:
    * 0,01
    * 0,03
    * 0,05
    * 0,07
    * 0,1
    * 0,2
*	Epochs:
    * 10
    * 20
    * 30
    * 50
    * 100
Propuestas de otros enfoques:
*	Probar separar 2 o más ítems por usuario para el test set.
*	Considerar la antigüedad de las compras.
*	Considerar un peso diferente para los artículos más comprados.
*	Tener en cuenta el precio de los productos.
*	Utilizar un learning rate dinámico.


Para ejecutar el código, se debe tener el fichero “HM_interactions.csv” en la misma carpeta. Para ejecutar el código no es necesario utilizar una GPU, pero se obtiene un mayor rendimiento en los tiempos de ejecución.
