from keras import layers,models

def cnnModel():
  input_yAv = layers.Input(shape=(10,5200)) # input shape 10x5200
  input_v = layers.Input(shape=(2,360,260))
  
  yAv = layers.Conv1D(20,3,data_format='channels_first',activation='relu',padding='same')(input_yAv) #20x5200
  yAv = layers.Conv1D(40,3,data_format='channels_first',activation='relu',padding='same')(yAv) #40x5200
  yAv = layers.MaxPooling1D(data_format='channels_first',pool_size=2)(yAv) #40x2600
  yAv = layers.Conv1D(80,3,data_format='channels_first',activation='relu',padding='same')(yAv) #80x2600
  yAv = layers.MaxPooling1D(data_format='channels_first',pool_size=2)(yAv) #80x1300
  yAv = layers.Conv1D(360,3,data_format='channels_first',activation='relu',padding='same')(yAv) #360x1300
  yAv = layers.MaxPooling1D(data_format='channels_first',pool_size=5)(yAv) #360x260
  yAv2D = layers.Reshape((1,360,260))(yAv) #1x360x260 
  yAv2D = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(yAv2D) #32x360x260
  yAv2D = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(yAv2D) #32x360x260
  
  v = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(input_v) #32x360x260
  v = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(v) #32x360x260
  v = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(v) #32x360x260
  
  H = layers.Concatenate(axis=1)([v,yAv2D]) #64x360x260
  H = layers.Conv2D(32,(3,3),data_format='channels_first',activation='relu',padding='same')(H)
  H = layers.Conv2D(16,(3,3),data_format='channels_first',activation='relu',padding='same')(H)
  H = layers.Conv2D(8,(3,3),data_format='channels_first',activation='relu',padding='same')(H)
  H = layers.Conv2D(4,(3,3),data_format='channels_first',activation='relu',padding='same')(H)
  H_prev = layers.Conv2D(2,(3,3),data_format='channels_first',activation='relu',padding='same')(H) #2x360x260
  ###### linear transformation to y-A(v)
  lt_yAv = layers.Conv1D(20,3,strides=1,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer1')(input_yAv) # 20x5200
  lt_yAv = layers.Conv1D(40,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer2')(lt_yAv) # 40x2600
  lt_yAv = layers.Conv1D(80,3,strides=2,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer3')(lt_yAv) # 80x1300
  lt_yAv = layers.Conv1D(360,3,strides=5,data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer4')(lt_yAv) # 360x260
  lt_yAv = layers.Reshape((1,360,260))(lt_yAv)
  lt_yAv = layers.Conv2D(2,(3,3),data_format='channels_first',padding='same',activation=None,use_bias=False,name='lt_layer_output')(lt_yAv) # 2x360x260
  #### pointwise multiplication
  def ptwsmultiply(x):
    H,lt_yAv = x
    H = H*lt_yAv
    return H

  H_out = layers.Lambda(ptwsmultiply)([H_prev,lt_yAv])
  H_tanh = layers.Conv2D(2,(1,1),data_format='channels_first',use_bias=False,activation='tanh',padding='same')(H_out)
  model = models.Model(inputs=[input_yAv,input_v],outputs=H_tanh)
  return model
