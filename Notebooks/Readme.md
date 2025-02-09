Modelos
---
- Unet_resize:
    - 100 epocas
    - val_score 0.7467, val_dice: 0.5932, test_score 0.7803 ,test_dice 0.5287
    - se pierden muchos detalles al hacer resize,aparecen muchas clases entremezcladas
    - al tener imagenes de distinto tamaño y resolucion parece ir peor en algunas
- Unet_crop:
    - 60 epocas, train 0.95
    - val_accuracy: 0.8393 , val_dice_multiv2: 0.5128, test_score 0.8764 ,test_dice 0.695
    - **Nota**: el dice en validacion aparentemente es peor, sin embargo esto es falso. Al ser recortes de img grandes 
    puede ser que en ellos aparezcan 2 o 3 clases en vez de las 6 que suelen aparecer en la img original. El dice score hace la media
    de todas las clases, si no hya, lo pone a cero y hace la media, por tanto, dando un resultado menor en tos casos. **Mejorar DICE**--> Hecho,
    aún así, al ser procesos en batches siempre van a aparecer todas las clases el problema reside en que la clase 5 (unlabeled) está infrarepresentada,1/10
    menos que otras clases.     
Si por ejemplo se eliminase esta clase y pasasemos de 6 clases a 5, el dice en validacion aumentaria a 0.61 y en test a 0.83. Ver [Anexo](Anexo). Además, esto se corrobora para predecciones de un unico elemento, donde el dice suele ser elevado.
    
    - La accuracy en validacion es mucho mejor, en test tanto accuracy como dice son bastante mejores que el caso anterior.
    - validacion estabilizada entorno a las 20-40 epocas.
- Deeplab v2: 
    - 15 epocas, 20min, train 0.95
    - val_accuracy: 0.8366 , test_score 0.8966,dice test: 0.7141
    - coste computacional mayor ~1.5/2 veces mas
    - resultados iguales o mejores con menos epocas -> coste computacional final menor
    - val_accuracy similar, resultados en test mejores, un 2% cada métrica
    - validacion no estabilizada, se podria sacar mas con alguna epoca extra. Hay un pico anomalo en el validation_loss
- Deeplab v1: 
    - 15 epocas, 25min, train 0.94
    - val_accuracy: 0.7973 , test_score 0.8533 ,dice test: 0.693876
    - coste computacional mayor ~1.5/2 veces mas (Unet_crop). Ligero coste computacional mayor que Deeplab v2
    - resultados iguales y peores que Deeplab v2
    - validacion no estabilizada, se podria sacar mas con alguna epoca extra. Hay varios picos anomalos en el validation_loss y con mas oscilaciones.
    - Hay epocas previas con mejores precisiones, se poría obtener estas o aumentar el numero de epocas.


Anexo
---
La media se define como:        
```math
M_{N}=\frac{\sum_{i=0}^{N}x_i}{N}
```
Para un elemento menos:
```math
M_{N-1}=\frac{\sum_{i=0}^{N-1}x_i}{N}
```
Dividiendo:
```math
frac{M_{N}}{M_{N-1}}=\frac{N-1}{N} \frac{\sum_{i=0}^{N}x_i}{\sum_{i=0}^{N}x_i}     
                    =\frac{N-1}{N} \frac{x_0+x_1+...+x_{N-1}+x_{N}}{x_0+x_1+...+x_{N-1}}      
                    =\frac{N-1}{N} \frac{a+x_{N}}{a}
                    =\frac{N-1}{N} (1-\frac{x_{N}}{a})   ,si x_{N}=0
                    =\frac{N-1}{N}       
M_{N-1}=\frac{N}{N-1}M_{N}
```