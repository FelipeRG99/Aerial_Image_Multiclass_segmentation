from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, MaxPool2D
from tensorflow.keras.models import Model
import tensorflow as tf

def double_conv_block(x, n_filters):

   # 2* Conv2D, ReLU activation
   x = Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)#, kernel_initializer = "glorot_normal"
   x = Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)

   return x
def encoder_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = MaxPool2D(pool_size =(2,2))(f)
   p = Dropout(0.1)(p)

   return f, p
def decoder_block(x, conv_features, n_filters):
   # upsample/decoder
   x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   x = concatenate([x, conv_features])#concatenate with enconder
   x = Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x
def custom_Unet(shape=(224,224,3),classes=1,activation='sigmoid',filters=[]):
   # inputs
   inputs = Input(shape=shape)

   # encoder: contracting path - downsample
   # 1 
   f1, p1 = encoder_block(inputs, filters[0])
   # 2 
   f2, p2 = encoder_block(p1, filters[1])
   # 3 
   f3, p3 = encoder_block(p2, filters[2])
   # 4 
   f4, p4 = encoder_block(p3, filters[3])

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, filters[4])

   # decoder: expanding path - upsample
   # 6 
   u6 = decoder_block(bottleneck, f4, filters[5])
   # 7
   u7 = decoder_block(u6, f3, filters[6])
   # 8
   u8 = decoder_block(u7, f2, filters[7])
   # 9 
   u9 = decoder_block(u8, f1, filters[8])

   # outputs
   outputs = Conv2D(classes, 1, padding="same", activation = activation)(u9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model