# Reconocimiento facial para el ingreso a espacios en una universidad - Grupo 3 - Taller de Proyectos Tecnológicos 

Este repositorio contiene los archivos necesarios para ejecutar un sistema de reconocimiento facial para el ingreso a espacios en una universidad. A continuación se detallan consideraciones a tener en cuenta al momento de su ejecución:

* En los scripts mk_usr.py y model.py, se debe cambiar $path\_db$ (líneas 18 y 56 respectivamente) por el directorio donde se estará la Base de Datos con las imágenes de los usuarios registrados.

* Para incorporar un usuario a la Base de Datos, se debe ejecutar el script mk_usr.py donde se debe ingresar el nombre de la persona para comenzar con el escaneo facial y luego proceder con el reentrenamiento del modelo.

* Para hacer funcionar el sistema, se debe ejecutar el script model.py. Las condiciones en que este se utilice, ya sea la luminosidad o distancia en la que se ubiquen las personas, puede afectar de manera considerable los resultados obtenidos. Es por esto que se debe sintonizar un umbral adecuado para la detección de los usuarios registrados. Inicialmente, una vez se entre a la etapa de verificación de usuario luego de haber detectado el movimiento, en consola se mostrará el valor actual del umbral y el valor detectado. Haciendo pruebas con otros usuarios, se debe ajustar este valor presente en la línea 33 del script.