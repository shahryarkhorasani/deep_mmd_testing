import keras
from keras.layers import Activation, Conv3D, LeakyReLU, PReLU, Add, Conv3DTranspose

def ActivationOp(layer_in, activation_type, name=None, shared_axes=[1, 2, 3], l=0.1):
    if (activation_type != 'prelu') & (activation_type != 'leakyrelu'):
        return Activation(activation_type, name=name)(layer_in)
    elif activation_type == 'prelu':
        return PReLU(alpha_initializer=keras.initializers.Constant(value=l), shared_axes=shared_axes, name=name)(
            layer_in)
    else:
        # TODO: check if alpha should be 0.01 instead
        return LeakyReLU(l)(layer_in)

def ResidualBlock3D(layer_in, depth=3, kernel_size=2, filters=None, activation='relu', kernel_initializer='he_normal',
                    name=None):
    # creates a residual block with a given depth for 3D input
    # there is NO non-linearity applied to the output! Has to be added manually
    l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer='he_normal',
               name='{}_c0'.format(name))(layer_in)
    for i in range(1, depth):
        a = ActivationOp(l, activation, name='{}_a{}'.format(name, i - 1))
        l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer='he_normal',
                   name='{}_c{}'.format(name, i))(a)
    o = Add()([layer_in, l])
    # o = Activation_wrap(o, activation, name='{}_a{}'.format(name,depth))
    return o 


def DownConv3D(layer_in, kernel_size=2, strides=(2, 2, 2), filters=None, activation='relu',
               kernel_initializer='he_normal', name=None):
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    dc = Conv3D(filters, kernel_size, strides=strides, padding='valid', activation='linear',
                name='{}_dc0'.format(name), kernel_initializer=kernel_initializer)(layer_in)
    dc = ActivationOp(dc, activation, name='{}_a0'.format(name))
    return dc

def UpConv3D(layer_in, kernel_size=(2, 2, 2), strides=None, filters=None, activation='relu',
             kernel_initializer='he_normal', name=None,
             data_format=None):
    if strides is None:
        strides = kernel_size
    elif isinstance(strides, int):
        strides = (strides, strides, strides)
    uc = Conv3DTranspose(filters, kernel_size=kernel_size, strides=strides, activation='linear',
                         name='{}_uc0'.format(name), kernel_initializer=kernel_initializer, padding='same')(layer_in)
    uc = ActivationOp(uc, activation, name='{}_a0'.format(name))
    return uc
