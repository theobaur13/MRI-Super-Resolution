# Code and models taken from https://github.com/razeineldin/DeepSeg

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Activation, Reshape, Permute, MaxPooling2D, Dropout
IMAGE_ORDERING = 'channels_last'

# UNet encoder
def relu6(x):
    return K.relu(x, max_value=6)

def get_unet_encoder(input_height=224,  input_width=224, depth=3, filter_size = 32, kernel = 3, pool_size = 2, encoder_name='UNet'):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(depth, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, depth))
    # 64
    conv1 = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(img_input)
    conv1 = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool1)
    conv2 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 256
    conv3 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool2)
    conv3 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 512
    conv4 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool3)
    conv4 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 1024
    conv5 = Conv2D(filter_size*2**4, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool4)
    conv5 = Conv2D(filter_size*2**4, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv5)
    drop5 = Dropout(0.5)(conv5)

    return img_input, [conv1, conv2, conv3, conv4, drop5]

# UNet decoder
def get_decoder_model(input, output):
    img_input = input
    o = output
    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height * output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height * output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.encoder_name= ""
    return model

def get_unet_decoder(n_classes, encoder, input_height=224, input_width=224, depth=3, filter_size=32, encoder_name=None, up_layer=False, trainable=True):
    img_input, levels = encoder(input_height=input_height, input_width=input_width, depth=depth, filter_size=filter_size, encoder_name=encoder_name)
    [f1, f2, f3, f4, f5] = levels 
     
    # 512
    up6 = Conv2D(filter_size*2**3, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(f5))
    merge6 = concatenate([f4,up6], axis = 3)
    conv6 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge6)
    conv6 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv6)
    # 256
    up7 = Conv2D(filter_size*2**2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([f3,up7], axis = 3)
    conv7 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge7)
    conv7 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv7)
    # 128
    up8 = Conv2D(filter_size*2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([f2,up8], axis = 3)
    conv8 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge8)
    conv8 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv8)
    # 64
    up9 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([f1,up9], axis = 3)
    o = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge9)
    o = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    if(up_layer):
        up10 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(o))
        merge10 = concatenate([img_input,up10], axis = 3)
        o = Conv2D(int(filter_size/2), (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge10)
        o = Conv2D(int(filter_size/2), (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    o = Conv2D(n_classes, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    model = get_decoder_model(img_input, o)
    if (encoder_name == 'ResNet50') or (encoder_name == 'NASNetMobile') or (encoder_name == 'NasNetLarge'):
        n_pads = 2 # for one_side_pad
    elif encoder_name == 'Xception':
        n_pads = 5 # for 3 one_side_pad
    else:
        n_pads = 0 # for reshape

    if not trainable and up_layer:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-26-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    elif not trainable:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-22-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    return model

def construct_unet(path):
    n_classes = 2
    input_height = 240
    input_width = 240
    depth = 3
    filter_size = 32
    encoder_name = 'UNet'

    encoder = get_unet_encoder
    model = get_unet_decoder(n_classes, encoder, input_height=input_height, input_width=input_width, depth=depth, filter_size=filter_size, encoder_name=encoder_name, up_layer=False, trainable=False)
    model.load_weights(path)

    return model