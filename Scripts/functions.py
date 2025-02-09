import json
import numpy as np
from tqdm import tqdm
import os
import cv2
from tensorflow.keras import backend as tf_backend
import tensorflow as tf
import matplotlib.pyplot as plt




#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def from_hex2rgb(hex_color):
    # Eliminar el carácter '#' si está presente
    hex_color = hex_color.lstrip('#')
    # Convertir cada par de caracteres hexadecimales en un valor RGB
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
def from_rgb2bgr(color):
    return np.array([color[2],color[1],color[0]])
def color_to_2D_label(msk,type_='bgr',dict_clases={}):
    """
    FUNCION: 
        Pasar de mascara recien cargada a labels/clases
    PARAMS:
        msk: np.array de tamaño (y,x,3)
        type_: tipo de transformacion de bgr o rgb
    RETURN:
        label_seg: np.array de tamaño (y,x) de labels
    """
    label_seg = np.zeros(msk.shape,dtype=np.uint8)
    for i in dict_clases:
        label_seg [np.all(msk == dict_clases[i][type_],axis=-1)] = dict_clases[i]['value']

    label_seg = label_seg[:,:,0]

    return label_seg
def label_to_color(pred_mask,type_='bgr',dict_clases={}):
    """
    FUNCION: 
        Pasar de mascaras con clase/label asignada a color
    PARAMS:
        pred_mask: np.array de tamaño (y,x)
        type_: tipo de transformacion a bgr o rgb
    RETURN:
        label_seg: np.array de tamaño (y,x,3) de colores
    """
    label_seg = np.zeros((pred_mask.shape[0],pred_mask.shape[0],3),dtype=np.uint8)
    for i in dict_clases:
        label_seg [np.where(pred_mask == dict_clases[i]['value'])] = dict_clases[i][type_]
    return label_seg
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def open_imgs(file='../Img/Tile 1/images/image_part_002.jpg',resize_=False,data_path_images='',resize_shape=(224,224)):
    """
    FUNCION: 
        cargar las img y sus msk, ademas, aplica un resize si se quiere
        
    PARAMS:
        file: directorio donde estan las img
        resize_: bool para indicar si se hace resiza
        data_path_images: directorio donde estan las img
        resize_shape: tupla (y,x) indicando el tamño del resize

    RETURN:
        resize_img,resize_msk: np.array de las img o msk
    """
    img=cv2.imread(file)
    tile=file.split('/')[2]
    name_img=file.split('/')[-1].replace('.jpg','')
    img_msk=cv2.imread(f'{data_path_images}/{tile}/masks/{name_img}.png')
    if resize_==True:
        resize_img=cv2.resize(img,resize_shape)
        resize_msk=cv2.resize(img_msk,resize_shape)
    else:
        resize_img,resize_msk=[img,img_msk]

    return resize_img,resize_msk
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def search_dir(data_path_images=''):
    """
    FUNCION: 
        obtener directorios de las img
        
    PARAMS:
        data_path_images: directorio donde estan las img

    RETURN:
        dire_end: lista con directorios de las imagenes

    """
    tiles=os.listdir(data_path_images)
    dire_end=[]
    for tile in tiles:
        dire=os.listdir(f'{data_path_images}/{tile}/images/')
        dire=[f'{data_path_images}/{tile}/images/'+file for file in dire]
        dire_end+=dire
    return dire_end
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def process_img(resize_=False,resize_shape=(224,224),dict_clases={},data_path_images=''):
    """
    FUNCION: 
        Procesar todas las imagenes en el directorio data_path_images
        
    PARAMS:
        resize_: bool para indicar si se hace resiza
        resize_shape: tupla (y,x) indicando el tamño del resize
    RETURN:
        labels,all_img: lista si resize_=False de np.arrays de las img o msk
                        np.array si resize_=True de tamaño (numero de imagenes,resize_shape,.) de las img o msk
    """
    #buscar paths
    filenames=search_dir(data_path_images=data_path_images)
    all_img=[]
    all_msk=[]
    ############# LOOP ABRIR/RESIZE IMG #################
    for file in tqdm(filenames):
        resize_img,resize_msk=open_imgs(file=file,resize_=resize_,resize_shape=resize_shape,data_path_images=data_path_images)
        all_img.append(resize_img)
        all_msk.append(resize_msk)
    if resize_==True:
        all_img=np.array(all_img)
        all_msk=np.array(all_msk)
    ############# LOOP TRANSFORM MASK TO LABELS ##############
    labels = []
    for mask in tqdm(all_msk):
        label = color_to_2D_label(mask,type_='bgr',dict_clases=dict_clases)
        labels.append(label)
    
    if resize_==True:
        labels = np.array(labels)
        print('Resize aplicado')
        print(labels.shape,all_img.shape)
    else:
        print(len(labels),len(all_img))

    return labels,all_img
SMOOTH = 1e-5
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def dice_multiv2(gt, pr, beta=1, num_classes=6, smooth=SMOOTH):
    """
    FUNCION: 
        Métrica de evaluacion coeficiente Dice (F1 score) para usar al entrenar modelos como metrica
    PARAMS:
        gt: Tensor ground truth (,y,x)
        pr: Tensor prediccion (,y,x,num_classes)
        beta: coeficiente para balancear precision y recall (beta=1 formula normal)
        num_classes: numero de clases a predecir
        smooth: numero para evitar divisiones por cero
    RETURN:
        Imagen y máscara recortadas.
    """


    # Convertir predicciones a clases discretas
    pr = tf_backend.argmax(pr, axis=-1)#pr(probabilidades)---> indice del mejor valor
    pr = tf_backend.one_hot(pr, num_classes=num_classes)#pr(indice del mejor valor)--->one hot
    gt = tf_backend.one_hot(tf_backend.cast(gt, tf.int32), num_classes=num_classes)
    # Convertir a float32 para evitar errores de tipo
    pr = tf_backend.cast(pr, tf.float32)
    gt = tf_backend.cast(gt, tf.float32)

    # Calcular intersección y unión (vectorizado)---> rapido
    intersection = tf_backend.sum(gt * pr, axis=[0, 1, 2])
    sum_gt = tf_backend.sum(gt, axis=[0, 1, 2])
    sum_pr = tf_backend.sum(pr, axis=[0, 1, 2])

    # Calcular TP, FP, FN
    tp = intersection #true positives
    fp = sum_pr - intersection# false positives
    fn = sum_gt - intersection# false negatives
    # F-beta score por clase (evitando divisiones por cero con smooth)
    fbeta_score_per_class = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)

    # Promedio de todas las clases
    overall_score = tf_backend.mean(fbeta_score_per_class)
    return overall_score
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def random_crop_image_mask(image, mask, crop_size=(128, 128)):
    """
    FUNCION: 
        Realiza un recorte aleatorio sincronizado entre una imagen y su máscara.
    PARAMS:
        image: Tensor de la imagen de entrada, tamaño (x, y, 3).
        mask: Tensor de la máscara correspondiente, tamaño (x, y).
        crop_size: Tamaño del recorte (alto, ancho).
    RETURN:
        Imagen y máscara recortadas.
    """
    combined = tf.concat([image, tf.expand_dims(mask, axis=-1)], axis=-1)#combinar img y msk
    cropped_combined = tf.image.random_crop(combined, size=[crop_size[0], crop_size[1], 4])#random crop  # 3 canales + 1 de máscara
    cropped_image = cropped_combined[..., :3]
    cropped_mask = tf.squeeze(cropped_combined[..., 3:], axis=-1)
    return cropped_image, cropped_mask
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def preprocess_list_with_crops(image_list, mask_list, crop_size=(128, 128),numer_crops=1):
    """
    FUNCION:
        Aplica random crops a listas de imágenes y máscaras.
    PARAMS:
        image_list: Lista de imágenes (x, y, 3).
        mask_list: Lista de máscaras (x, y).
        crop_size: Tamaño de los recortes.
        numer_crops: numero de cortes aleatorios realizados a una misma imagen
    RETURNS:
        Tensors con imágenes y máscaras recortadas.
    """
    cropped_images = []
    cropped_masks = []
    #################### LOOP IMG/MSK #####################
    for img, mask in tqdm(zip(image_list, mask_list)):
        #convertir a tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
        i=0
    ################# LOOP CROPS ###########################
        while i< numer_crops:
            #realizar cortes
            cropped_img, cropped_mask = random_crop_image_mask(img_tensor, mask_tensor, crop_size)
            #añadir a las listas
            cropped_images.append(cropped_img)
            cropped_masks.append(cropped_mask)
            i+=1

    return tf.stack(cropped_images), tf.stack(cropped_masks)
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def plot_images_comparation(list_images_rows,titles_rows=["Input Image", "Real Mask", "Predicted Mask"],save=False,filename=''):
    """
    FUNCION:    Plot comparativo de Img real, mascara y prediccion

    PARAMS:
        list_images_rows: lista de listas que contienen las imagenes [[img,msk,pred],[...],...]
        titles_rows: [] titulos de las imagenes
        save: bool indica si se guarda la imagen o no
        filename: str en caso de guardar indica el nombre

    RETURN:
        PLOT o imagen

    """
    num_rows = len(list_images_rows)
    num_cols=len(list_images_rows[0])
    if num_cols==2:
        titles_rows=["Input Image", "Predicted Mask"]
    fig=plt.figure(figsize=(15, 5*num_rows))
    index=1
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows,num_cols,index)
            plt.imshow(list_images_rows[i][j])
            plt.axis('off')
            if i==0:
                plt.title(titles_rows[j])  
            index+=1  
    plt.show()
    if save and filename!='':
        fig=plt.figure(figsize=(15, 5*num_rows))
        index=1
        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows,num_cols,index)
                plt.imshow(list_images_rows[i][j])
                plt.axis('off')
                if i==0:
                    plt.title(titles_rows[j])  
                index+=1  
        plt.savefig(filename, dpi=300, bbox_inches='tight')
def reconstruir_imagen(cut_img,tamano_original,tipo='img'):
    """
    FUNCION: Reconstruir una imagen orginal de tamaño (tamaño original(y,x))  a partir de sus cortes de tamaño (cut,cut)  

    PARAMS:
        trozos_imagen: np.array de los cortes (numero de cortes,cut,cut,channel)
        tamaño_original: tupla indicando el tamño de x e y originales
        type: indica si es una imagen o su mascara (msk), las mascaras no tienen la shape adicional de channel y las imagenes si

    RETURN:
        recon_img: np.array(y,x,channels) imgen reconstruida

    """
    cut=cut_img.shape[1]#obtener tamaño d elos cortes
    shape_original_y,shape_original_x=tamano_original#obtener tamño de la imagen original
    num_cuts_eje_x=(shape_original_x//cut)+1#obtener numero de cortes realizados en el eje x
    num_cuts_eje_y=(shape_original_y//cut)+1#obtener numero de cortes realizados en el eje y
    recon_img=np.array([])
    ##################### ASIGNAR TAMAÑO A IMAGEN FINAL ##################
    if tipo == 'img':
        channels=cut_img.shape[3]
        recon_img=np.zeros((shape_original_y,shape_original_x,channels))
    elif tipo=='msk':
        if len(cut_img.shape)==3:
            recon_img=np.zeros((shape_original_y,shape_original_x))
        elif len(cut_img.shape)==4:
            recon_img=np.zeros((shape_original_y,shape_original_x,1))
    index_tot=0
    #####################   LOOP EJE x ######################
    for i in range(num_cuts_eje_x):
        if i!=num_cuts_eje_x-1:
            cut_ini_x=(i)*cut  #corte inicial
            cut_fin_x=(i+1)*cut#corte final
        # PARA EL DESFASE DEL ULTIMO CORTE
        else:
            cut_ini_x=shape_original_x-cut
            cut_fin_x=shape_original_x
    #####################   LOOP EJE y ######################
        for j in range(num_cuts_eje_y):
            if j!=num_cuts_eje_y-1:
                cut_ini_y=(j)*cut#corte inicial
                cut_fin_y=(j+1)*cut#corte final
            # PARA EL DESFASE DEL ULTIMO CORTE
            else:
                cut_ini_y=shape_original_y-cut
                cut_fin_y=shape_original_y
    #####################   RECONSTRUIR IMAGEN ######################
            if tipo == 'img':
                recon_img[cut_ini_y:cut_fin_y,cut_ini_x:cut_fin_x,:]=cut_img[index_tot]
            elif tipo=='msk':
                if len(cut_img.shape)==3:
                    recon_img[cut_ini_y:cut_fin_y,cut_ini_x:cut_fin_x]=cut_img[index_tot]
                if len(cut_img.shape)==4:
                    recon_img[cut_ini_y:cut_fin_y,cut_ini_x:cut_fin_x,:]=cut_img[index_tot]

            index_tot+=1
    return recon_img
#################################################################################################################################
#######################################                             #############################################################
#################################################################################################################################
def obtener_corte(img,cut=224,tipo='img'):
    """
    FUNCION:    Obtener cortes de una imagen dada de un tamaño (cutxcut)

    PARAMS:
        img: np.array de la imagen por cv2.imread()
        cut: tamaño del corte
        type: indica si es una imagen o su mascara (msk)

    RETURN:
        array_tot: np.array(numero_cortes,cut,cut,canales) donde canales son 3 en las imagenes
        normales y nada en las mascaras

    """
    # Obtener las dimensiones de la imagen (height, width, channels)
    if tipo == 'img':
        height, width, channels = img.shape
        format=(1,cut,cut,channels)
    elif tipo == 'msk':
        height, width= img.shape
        format=(1,cut,cut)

    # Definir las coordenadas para el recorte
    max_val_cut_y=height-cut
    max_val_cut_x=width-cut
    array_tot=np.array([])
    ######################## LOOP EJE X #####################
    for j in range(1,int(width/cut)+2):
        if cut*j<width:
            cut_ini_x=cut*(j-1)#corte inicial
            cut_fin_x=cut*j#corte final
        # PARA EL DESFASE DEL ULTIMO CORTE
        else:
            cut_ini_x=max_val_cut_x
            cut_fin_x=width
    ########################### LOOP EJE Y ################
        for i in range(1,int(height/cut)+2):
            if cut*i<height:
                cut_ini=cut*(i-1)#corte inicial
                cut_fin=cut*i#corte final
            # PARA EL DESFASE DEL ULTIMO CORTE
            else:
                cut_ini=max_val_cut_y
                cut_fin=height
    #####################   REALIZAR CORTES ######################
            if tipo == 'img':
                cut_img = img[cut_ini:cut_fin, cut_ini_x:cut_fin_x,:]
            elif tipo == 'msk':
                cut_img = img[cut_ini:cut_fin, cut_ini_x:cut_fin_x]
            #agrupar imagenes en un mismo array
            if array_tot.shape[0]>0:
                array_tot=np.vstack([array_tot,cut_img.reshape(format)])
            else:
                array_tot=cut_img.reshape(format)
    return array_tot