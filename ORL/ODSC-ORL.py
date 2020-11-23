
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
#from numpy.linalg import svd
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
#import matlab.engine
# tf.train.Saver(max_to_keep=None) 

def next_batch(data, _index_in_epoch ,batch_size , _epochs_completed):
    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        #label = label[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/models_face/logs', no= 0):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        self.no = 0
        #input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        
        weights = self._initialize_weights()
        
        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        
        #self.Coef = tf.Variable(np.eye(batch_size,batch_size,0,np.float32))    
        z = tf.reshape(latent, [batch_size, -1])  
        Coef = weights['Coef']         
        z_c = tf.matmul(Coef,z)    
        self.Coef = Coef       
        latent_c = tf.reshape(z_c, tf.shape(latent)) 
        self.z = z       
        
        self.x_r = self.decoder(latent_c, weights, shape)                
        
        # l_2 reconstruction loss 
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost)
       
        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef,2.0))
        #reg_constant3 = 1.0
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses )
        
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        #re_constant2 = 1.0
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  
        
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)        
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))],max_to_keep=None) 
        #[v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]       
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef']   = tf.Variable(0 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')        
        
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32)) # , name = 'enc_b0'
        
        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i-1], \
                        self.n_hidden[iter_i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(tf.zeros([self.n_hidden[iter_i]], dtype = tf.float32)) # , name = enc_name_bi
            iter_i = iter_i + 1
        
        iter_i = 1
        while iter_i < n_layers:    
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers-iter_i], self.kernel_size[n_layers-iter_i], 
                        self.n_hidden[n_layers-iter_i-1],self.n_hidden[n_layers-iter_i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(tf.zeros([self.n_hidden[n_layers-iter_i-1]], dtype = tf.float32)) # , name = dec_name_bi
            iter_i = iter_i + 1
            
        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype = tf.float32)) # , name = dec_name_bi
        
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())
        
        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1,2,2,1],padding='SAME'),weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1
        
        layer3 = layeri
        return  layer3, shapes
    
    # Building the decoder
    def decoder(self,z, weights, shapes):
        n_layers = len(self.n_hidden)        
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            #if iter_i == n_layers-1:
            #    strides_i = [1,2,2,1]
            #else:
            #    strides_i = [1,1,1,1]
            shape_de = shapes[n_layers - iter_i - 1]            
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0],shape_de[1],shape_de[2],shape_de[3]]),\
                     strides=[1,2,2,1],padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3
    
    def partial_fit(self, X, lr): #  
        cost, summary, _, Coef = self.sess.run((self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef), feed_dict = {self.x: X, self.learning_rate: lr})#
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        return self.sess.run(self.x_r, feed_dict = {self.x:X})
    
    def transform(self, X):
        return self.sess.run(self.z, feed_dict = {self.x:X})
    
    def save_model(self):
        self.no = self.no+1
        savetmp = self.model_path + "%d.ckpt"%(self.no)
        # save_path = self.saver.save(self.sess,self.model_path)
        save_path = self.saver.save(self.sess, savetmp)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")
 
 
class newConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, learning_rate = 0.001,batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = './mymodels/logs', no=0):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        self.no = 0 
        
        #input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        
        weights = self._initialize_weights()
        
        if denoise == False:
            x_input = self.x
            latent, pool1, shape = self.encoder(x_input, weights)
            # olatent, oshape = self.encoderover(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        # print("HERE")
        print(latent.shape,pool1.shape)

        latent = tf.add(latent,pool1)
        z = tf.reshape(latent, [batch_size, -1])  
        # z2 = tf.reshape(laten2, [batch_size, -1])  

        Coef = weights['Coef']     
        # Coef2 = weights['oCoef']    

        z_c = tf.matmul(Coef,z)    
        # z_c2 = tf.matmul(Coef2,z2)  

        self.Coef = Coef        

        latent_c = tf.reshape(z_c, tf.shape(latent)) 
        # latent_c2 = tf.reshape(z_c2, tf.shape(laten2)) 
        # print(z.shape)
        self.z = z       
        # print(z.shape)
        
        self.x_r = self.decoder(latent_c,  weights, shape)                
        
        # l_2 reconstruction loss 
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        
        tf.summary.scalar("recons_loss", self.reconst_cost)
                
        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef,2.0))
        
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses )
        
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  
        
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)        
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])              
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        
        
        all_weights['Coef']   = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')
        
        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
        
        all_weights['oenc_w0'] = tf.get_variable("oenc_w0", shape=[self.kernel_size[0], self.kernel_size[0],  self.n_hidden[0],1],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['oenc_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))

        all_weights['oenc_w1'] = tf.get_variable("oenc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[2],self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['oenc_b1'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))

        all_weights['oenc_w2'] = tf.get_variable("oenc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[2],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['oenc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        
        
        all_weights['oCoef']   = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')
        
        all_weights['odec_w0'] = tf.get_variable("odec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['odec_b0'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))

        all_weights['odec_w1'] = tf.get_variable("odec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['odec_b1'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))

        all_weights['odec_w2'] = tf.get_variable("odec_w2", shape=[self.kernel_size[0], self.kernel_size[0],3, 3],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['odec_b2'] = tf.Variable(tf.zeros([3], dtype = tf.float32))

        all_weights['odec_w3'] = tf.get_variable("odec_w3", shape=[self.kernel_size[0], self.kernel_size[0],3, 3],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['odec_b3'] = tf.Variable(tf.zeros([3], dtype = tf.float32))
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)

        shapes_en = shapes[0]

        olayer1 = tf.add(tf.nn.conv2d_transpose(x, weights['oenc_w0'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*2,64,3]),\
            strides=[1,2,2,1],padding='SAME'),weights['oenc_b0'])
        olayer1 = tf.nn.relu(olayer1)
        # shapes.append(layer1.get_shape().as_list())
        olayer2 = tf.add(tf.nn.conv2d_transpose(olayer1, weights['oenc_w1'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*4,shapes_en[2]*4,6]),\
         strides=[1,2,2,1],padding='SAME'),weights['oenc_b1'])
        olayer2 = tf.nn.relu(olayer2)

        pool1 = tf.layers.max_pooling2d(inputs=olayer2, pool_size=[2,2], strides=32)
        # pool2 = tf.layers.max_pooling2d(inputs=olayer2, pool_size=[2,2], strides=32)
        
        # olayer3 = tf.add(tf.nn.conv2d(pool1, weights['odec_w2'],strides=[1,4,4,1],padding='SAME'),weights['odec_b2'])
        # olayer3 = tf.nn.relu(olayer3)

        # olayer4 = tf.nn.conv2d(pool1, weights['odec_w3'], strides=[1,2,2,1],padding='SAME')
        # olayer4 = tf.nn.relu(olayer4)
        
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)

        # print(layer2.shape,olayer4.shape)
        # layer2 = tf.add(layer2,pool1)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)

        # print(layer3.shape,pool2.shape)
        # layer3 = tf.add(layer3,pool2)


        return  layer3, pool1, shapes #add olayer2 as 2nd arg
    
    # Building the decoder
    def decoder(self,z,  weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0],shape_de2[1],shape_de2[2],shape_de2[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0],shape_de3[1],shape_de3[2],shape_de3[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        # print(layer3.shape, layer2.shape)

        # olayer1 = tf.add(tf.nn.conv2d(z2, weights['odec_w0'],strides=[1,2,2,1],padding='SAME'),weights['odec_b0'])
        # olayer1 = tf.nn.relu(olayer1)
        # shape_de2 = shapes[1]
        # olayer2 = tf.add(tf.nn.conv2d(olayer1, weights['odec_w1'],strides=[1,2,2,1],padding='SAME'),weights['odec_b1'])
        # olayer2 = tf.nn.relu(olayer2)
        
        # print(layer3.shape, olayer1.shape, olayer2.shape)
        
        # layer3 = tf.add(layer3,olayer2)

        
        return layer3

    
    
    def partial_fit(self, X, lr): #  
        cost, summary, _, Coef = self.sess.run((self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef), feed_dict = {self.x: X, self.learning_rate: lr})#
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        print(self.x_r.shape)
        print(self.x.shape)
        return self.sess.run(self.x_r, feed_dict = {self.x:X})
    
    def transform(self, X):
        return self.sess.run(self.z, feed_dict = {self.x:X})
    
    def save_model(self):
        self.no = self.no+1
        savetmp = self.model_path + "%d.ckpt"%(self.no)
        # save_path = self.saver.save(self.sess,self.model_path)
        save_path = self.saver.save(self.sess, savetmp)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")



def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)      
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)    
    U = U.dot(S)    
    U = normalize(U, norm='l2', axis = 1)       
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha) 
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 
        
        
def test_face(Img, Label, CAE, num_class):       
    
    alpha = 0.19
    # print alpha
    
    acc_= []
    for i in range(0,41-num_class): 
        face_10_subjs = np.array(Img[10*i:10*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[10*i:10*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model    
        
        max_step = 800 #50 + num_class*25# 100+num_class*20
        display_step = 5 #10
        lr = 0.001
        # fine-tune network
        epoch = 0
        best = 0
        while epoch < max_step:
            epoch = epoch + 1           
            cost, Coef = CAE.partial_fit(face_10_subjs, lr)#                                  
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(batch_size)))                
                Coef = thrC(Coef,alpha)                                                       
                y_x, _ = post_proC(Coef, label_10_subjs.max(), 3,1)                  
                missrate_x = err_rate(label_10_subjs, y_x)                
                acc_x = 1 - missrate_x 
                # print("experiment: %d" % i, "our accuracy: %.4f" % acc_x)
                if acc_x>best:
                    best = acc_x
                    # print("BEST", epoch)
        acc_.append(best)    
    
    acc_ = np.array(acc_)
    m = np.mean(acc_)
    me = np.median(acc_)
    mean_t = ((1-m)*100)
    # if best_m>mean_t:
    #     best_m = mean_t

    # print("%d subjects:" % num_class)    
    # print("Mean: %.4f%%" % ((1-m)*100))    
    # print("Median: %.4f%%" % ((1-me)*100))
    # # print(acc_) 

    # print("BEST error: %.4f%%" % mean_t)
    
    return (1-m), (1-me)
def test_facep(Img, CAE, n_input):
    # print(Img.shape)
    batch_x_test = Img[0:200,:]
    # print(batch_x_test.shape)
    # batch_x_test= np.reshape(batch_x_test,[100,n_input[0],n_input[1],1])
    # print(batch_x_test.shape)
    CAE.restore()
    x_re = CAE.reconstruct(batch_x_test)

    plt.figure(figsize=(8,12))
    for i in range(5):
        plt.subplot(5,2,2*i+1)
        plt.imshow(batch_x_test[i,:,:,0], vmin=0, vmax=255, cmap="gray") #
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_re[i,:,:,0], vmin=0, vmax=255, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    plt.show()
    return
def train_face(Img, CAE, n_input, batch_size):    
    it = 0
    display_step = 1
    save_step = 1
    _index_in_epoch = 0
    _epochs= 0
    lr = 0.01
    # print("IN")
    # CAE.restore()
    # train the network
    while True:
        batch_x,  _index_in_epoch, _epochs =  next_batch(Img, _index_in_epoch , batch_size , _epochs)
        batch_x = np.reshape(batch_x,[batch_size,n_input[0],n_input[1],1])
        cost = CAE.partial_fit(batch_x, lr)
        it = it +1
        # print(cost,batch_size)
        avg_cost = cost[0]/batch_size
        # if it % display_step == 0:
            # print ("epoch: %.1d" % _epochs)
            # print  ("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
            
            # test_facep(Img, CAE, n_input)
            break
    return
        
best_m = 100  
if __name__ == '__main__':
    
    # load face images and labels
    data = sio.loadmat('.././Data/ORL_32x32.mat')
    Img = data['fea']
    Label = data['gnd']     
    
    # face image clustering
    n_input = [32, 32]
    kernel_size = [3,3,3]
    n_hidden = [3, 3, 6]
    
    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 
    
    all_subjects = [40]
    
    avg = []
    med = []
    
    iter_loop = 0
    tot_ep = 56
    i = 55
    best_m = 100
    testep = 1

    while i<tot_ep:
        i = i+1
        iter_loop = 0
        while iter_loop < len(all_subjects):
            num_class = all_subjects[iter_loop]
            batch_size = num_class * 10
            reg1 = 2.0
            reg2 = 0.1
            print ("epoch: %.1d" % i)
             

            model_path = './pretrained/modelover' + "%d.ckpt"%i
            # model_path = '.././mymodels/orl/modelover' + "%d.ckpt"%i 
            # restore_path = './pretrain-model-ORL/model-335-32x32-orl.ckpt'  
            # restore_path = './mymodels/orl/over/modelnew5.ckpt' 
            logs_path = './ft/logs' 
            tf.reset_default_graph()
            CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                         kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=model_path, logs_path=logs_path)
            
            # train_face(Img, CAE, n_input, batch_size)

            # CAE.restore()
            if i%testep ==0:
                # test_facep(Img, CAE, n_input)
                avg_i, med_i = test_face(Img, Label, CAE, num_class)
                mean_t = avg_i
                print(mean_t)
                if best_m>mean_t:
                    best_m = mean_t
                    bestep = i


                print("BEST error: %.4f%% at epoch %d" % (best_m,bestep))
                avg.append(avg_i)
                med.append(med_i)
            iter_loop = iter_loop + 1
            
        iter_loop = 0
        # while iter_loop < len(all_subjects):
        #     num_class = all_subjects[iter_loop]
        #     print('%d subjects:' % num_class)
        #     print('Mean: %.4f%%' % (avg[iter_loop]*100), 'Median: %.4f%%' % (med[iter_loop]*100)) 
        #     iter_loop = iter_loop + 1      
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
