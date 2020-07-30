from keras import layers,models
from keras.utils import multi_gpu_model

def residual_stack1D(x,n_chann=32, downsize=2):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv1D(n_chann, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv1D(n_chann, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  x = layers.Conv1D(n_chann, data_format='channels_first',kernel_size=1, padding='same',activation='linear',kernel_initializer="he_normal")(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  # maxpool for down sampling
  x = layers.MaxPooling1D(data_format='channels_first',pool_size=downsize)(x)
  return x

def residual_stack2D(x,n_chann=32):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv2D(n_chann, kernel_size=(5,5),data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(n_chann, kernel_size=(5,5),data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  x = layers.Conv2D(n_chann, data_format='channels_first',kernel_size=(1,1), padding='same',activation='linear',kernel_initializer="he_normal")(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  return x

def ptwsmultiply(x):
  H,lt_yAv = x
  H = H*lt_yAv
  return H

def Model18():
  # Construct DPM neural network architecture
  input_yAv = layers.Input(shape=(10,5200)) # input shape 10x5200
  input_v = layers.Input(shape=(2,360,260)) 
  yAv = residual_stack1D(input_yAv, n_chann=20, downsize=1) # output shape 20x5200
  yAv = residual_stack1D(yAv, n_chann=40) # output shape 40x2600
  yAv = residual_stack1D(yAv, n_chann=80) # output shape 80 x 1300
  yAv = residual_stack1D(yAv, n_chann=360, downsize=5) # output shape 360 x 260
  yAv2D = layers.Reshape((1,360,260))(yAv)
  yAv2D = residual_stack2D(yAv2D)
  yAv2D = residual_stack2D(yAv2D)
  v = residual_stack2D(input_v)
  v = residual_stack2D(v)
  v = residual_stack2D(v)
  H = layers.Concatenate(axis=1)([v,yAv2D])
  H = residual_stack2D(H,n_chann=64)
  H = residual_stack2D(H,n_chann=32)
  H = residual_stack2D(H,n_chann=16)
  H = residual_stack2D(H,n_chann=8)
  H = residual_stack2D(H,n_chann=2)
  ###### linear transformation to y-A(v)
  lt_yAv = layers.Conv1D(20,3,strides=1,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer1')(input_yAv) # 20x5200
  lt_yAv = layers.Conv1D(40,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer2')(lt_yAv) # 40x2600
  lt_yAv = layers.Conv1D(80,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer3')(lt_yAv) # 80x1300
  lt_yAv = layers.Conv1D(360,3,strides=5,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer4')(lt_yAv) # 360x260
  lt_yAv = layers.Reshape((1,360,260))(lt_yAv)
  lt_yAv = layers.Concatenate(axis=1)([lt_yAv,lt_yAv])
  #### pointwise multiplication
  H_out = layers.Lambda(ptwsmultiply,name='prior_ptwspd')([H,lt_yAv])
  H_tanh = layers.Conv2D(2,(1,1),data_format='channels_first',use_bias=False,activation='tanh')(H_out)
  model = models.Model(inputs=[input_yAv,input_v],outputs=H_tanh)
  return model

def Model19():
  # Construct DPM neural network architecture
  input_yAv = layers.Input(shape=(10,5200)) # input shape 10x5200
  input_v = layers.Input(shape=(2,360,260)) 
  yAv = residual_stack1D(input_yAv, n_chann=20, downsize=1) # output shape 20x5200
  yAv = residual_stack1D(yAv, n_chann=40) # output shape 40x2600
  yAv = residual_stack1D(yAv, n_chann=80) # output shape 80 x 1300
  yAv = residual_stack1D(yAv, n_chann=360, downsize=5) # output shape 360 x 260
  yAv2D = layers.Reshape((1,360,260))(yAv)
  yAv2D = residual_stack2D(yAv2D)
  yAv2D = residual_stack2D(yAv2D)
  v = residual_stack2D(input_v)
  v = residual_stack2D(v)
  v = residual_stack2D(v)
  H = layers.Concatenate(axis=1)([v,yAv2D])
  H = residual_stack2D(H,n_chann=64)
  H = residual_stack2D(H,n_chann=32)
  H = residual_stack2D(H,n_chann=16)
  H = residual_stack2D(H,n_chann=8)
  H = residual_stack2D(H,n_chann=2)
  H_prev = layers.Conv2D(2,(1,1),data_format='channels_first',use_bias=False,activation='tanh',name="prev_output")(H)
  ###### linear transformation to y-A(v)
  lt_yAv = layers.Conv1D(20,3,strides=1,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer1')(input_yAv) # 20x5200
  lt_yAv = layers.Conv1D(40,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer2')(lt_yAv) # 40x2600
  lt_yAv = layers.Conv1D(80,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer3')(lt_yAv) # 80x1300
  lt_yAv = layers.Conv1D(360,3,strides=5,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer4')(lt_yAv) # 360x260
  lt_yAv = layers.Reshape((1,360,260))(lt_yAv)
  lt_yAv = layers.Conv2D(2,(3,3),data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer_output')(lt_yAv) # 2x360x26
  #### pointwise multiplication
  H_out = layers.Lambda(ptwsmultiply,name='prior_ptwspd')([H_prev,lt_yAv])
  H_tanh = layers.Conv2D(2,(1,1),data_format='channels_first',use_bias=False,activation='tanh')(H_out)
  model = models.Model(inputs=[input_yAv,input_v],outputs=H_tanh)
  return model

def Model24():
  # Construct DPM neural network architecture
  input_yAv = layers.Input(shape=(10,5200)) # input shape 10x5200
  input_v = layers.Input(shape=(2,360,260))
  yAv = residual_stack1D(input_yAv, n_chann=20, downsize=1) # output shape 20x5200
  yAv = residual_stack1D(yAv, n_chann=40) # output shape 40x2600
  yAv = residual_stack1D(yAv, n_chann=80) # output shape 80 x 1300
  yAv = residual_stack1D(yAv, n_chann=360, downsize=5) # output shape 360 x 260
  yAv2D = layers.Reshape((1,360,260))(yAv)
  yAv2D = residual_stack2D(yAv2D)
  yAv2D = residual_stack2D(yAv2D)
  v = residual_stack2D(input_v)
  v = residual_stack2D(v)
  v = residual_stack2D(v)
  H = layers.Concatenate(axis=1)([v,yAv2D])
  H = residual_stack2D(H,n_chann=64)
  H = residual_stack2D(H,n_chann=32)
  H = residual_stack2D(H,n_chann=16)
  H = residual_stack2D(H,n_chann=8)
  H = residual_stack2D(H,n_chann=2)
  H_prev = layers.Conv2D(2,(1,1),data_format='channels_first',use_bias=False,activation='tanh',name="prev_output")(H)
  ###### linear transformation to y-A(v)
  lt_yAv = layers.Conv1D(20,3,strides=1,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer1')(input_yAv) # 20x5200
  lt_yAv = layers.Conv1D(40,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer2')(lt_yAv) # 40x2600
  lt_yAv = layers.Conv1D(80,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer3')(lt_yAv) # 80x1300
  lt_yAv = layers.Conv1D(360,3,strides=5,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer4')(lt_yAv) # 360x260
  lt_yAv = layers.Reshape((1,360,260))(lt_yAv)
  lt_yAv = layers.Conv2D(2,(3,3),data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer_output')(lt_yAv) # 2x360x26
  #### pointwise multiplication
  H_out = layers.Lambda(ptwsmultiply,name='prior_ptwspd')([H_prev,lt_yAv])
  H_out = layers.Conv2D(32,(3,3),data_format='channels_first',padding='same',use_bias=False,activation=None)(H_out)
  H_out = layers.Conv2D(32,(3,3),data_format='channels_first',padding='same',use_bias=False,activation=None)(H_out)
  H_out = layers.Conv2D(32,(3,3),data_format='channels_first',padding='same',use_bias=False,activation=None)(H_out)
  H_tanh = layers.Conv2D(2,(3,3),data_format='channels_first',padding='same',use_bias=False,activation='tanh')(H_out)
  model = models.Model(inputs=[input_yAv,input_v],outputs=H_tanh)
  return model
