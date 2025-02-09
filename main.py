import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import json
import os

from Scripts.unet_architecture import custom_Unet
from Scripts.deeplab_architecture import custom_Deeplab
from Scripts.functions import from_hex2rgb, from_rgb2bgr, process_img,preprocess_list_with_crops,dice_multiv2,obtener_corte,reconstruir_imagen,color_to_2D_label,plot_images_comparation
model_type='unet'
crop_size=(224,224)

data_path_images='Img'
img_path='/Tile 8/images/image_part_002.jpg'
msk_path=img_path.replace('images','masks').replace('jpg','png')

data_path_images='main_img/img'
data_path_msk='main_img/mask'
img_path=os.listdir(data_path_images)[0]

save=True
################# Clases ###############################
with open("classes_corrected.json", "r") as archivo:
    datos_json = json.load(archivo)
dict_clases={}
i=0
for class_ in  datos_json['classes']:
    dict_clases[i]={'rgb':from_hex2rgb(class_['color']),'bgr':from_rgb2bgr(from_hex2rgb(class_['color'])),'name':class_['title'],'value':i}
    i+=1
################# MODELS ###############################
if model_type=='unet':
    model_load=custom_Unet(shape=crop_size+(3,),classes=6,activation='softmax',filters=[32,64,128,256,512,256,128,64,32])
    model_load.load_weights('Trained_Models/modelo-20-0.81.weights.h5')
    print('Unet')
elif model_type=='deeplab':
    model_load=custom_Deeplab(img_size=224,num_clases=6,activation='softmax')
    model_load.load_weights('Trained_Models/modelo-19-0.86_v2.weights.h5')
    print('Deeplab')
else:
    print('\033[31mModelo seleccionado incorrecto\nEscoge uno de los siguientes: unet, deeplab\033[0m')
    exit()
################ PRED #################################    
img=cv2.imread(data_path_images+'/'+img_path)
img_msk=cv2.imread(data_path_msk+'/'+img_path.replace('jpg','png'))
#realizar corte
trozos_imagen=obtener_corte(img)
#predicciones
preds=model_load.predict(trozos_imagen)
#reconstruyo la prediccion
pred_mask_recon=reconstruir_imagen(preds,(img.shape[0],img.shape[1]),tipo='img')
# Mostrar las imágenes en filas de tres
pred_binary=np.argmax(pred_mask_recon,axis=-1)
img_msk_label=color_to_2D_label(img_msk,type_='bgr',dict_clases=dict_clases)
plot_images_comparation([[img,img_msk_label,pred_binary]],save=save,filename='Imagenes resultados/'+model_type+'_'+img_path.replace('jpg','')+'.png')
print(f'Accuracy {accuracy_score(img_msk_label.flatten(),pred_binary.flatten())} : DICE {dice_multiv2(img_msk_label,pred_mask_recon).numpy()}')