*Análisis matemático II*\
\
*Trabajo práctico integrador*\
*Grupo 3*\
Integrantes: Felipe Albanesi, Guillermo Carranza.\
Comisión: 17-21\
Tema: _Ecuación de la órbita kepleriana_\
\
\
\
Fecha de entrega: 01-11-2026
\
\
\
Aspectos a tener en cuenta para correr el código: el código usa las librerías numpy (usada para calculo vectorial), pandas (para leer los archivos y los registros del mismo), matplotlib (para las visualizaciones) y pathlib para armar rutas a los archivos sin depender de desde dónde se ejecute.\
La librería pathlib es una librería estándar por lo que no debe descargarse. Las demás librerías pueden descargarse desde la terminal usnado el comando: pip install numpy pandas matplotlib. El comando puede cambiar dependiendo del sistema operativo, si no funciona se puede intentar con -m pip install numpy pandas matplotlib.\
\
\
En el código se añadió la posibilidad de elegir si usar el foco de la tabla o el de las observaciones en algunos casos. Por más que se indique en el programa, cabe resaltar que *el foco de la tabla es incosistente con las observaciones*, la elección se da con el fin de visualizar esto únicamente, lo correcto sería elegir el de las observaciones, que es el que se colocará por defecto en caso de que no se quiera elegir (que se ingrese cualquier dato que no sea "tabla" u "observaciones")