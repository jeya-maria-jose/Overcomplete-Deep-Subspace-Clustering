import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import matplotlib.pyplot as plt
# tf.train.Saver(max_to_keep=None) 


def next_batch(data, _index_in_epoch ,batch_size , _epochs_completed):
    _num_examples = data.shape[0]
    # print(_num_examples)
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


class overorgConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, learning_rate = 0.001, batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = './mymodels/over/logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        
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
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0],  self.n_hidden[0],1],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[1],self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[2],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        
        
        all_weights['Coef']   = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')
        
        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
        
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        # print(shapes)
        shapes_en = shapes[0]
        layer1 = tf.add(tf.nn.conv2d_transpose(x, weights['enc_w0'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*2,64,2]),\
            strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['enc_w1'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*4,shapes_en[2]*4,2]),\
         strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['enc_w2'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*8,shapes_en[2]*8,2]),\
         strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        # # print("encoder")
        # shapes.append(layer3.get_shape().as_list())
        return  layer3, shapes
    
    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        # shape_de1 = shapes[3]
        print(shapes)
        # print("decoder")
        # print(z.shape,weights['dec_w0'],weights['dec_b0'])
        layer1 = tf.add(tf.nn.conv2d(z, weights['dec_w0'],strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d(layer1, weights['dec_w1'],strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        
        layer2 = tf.nn.relu(layer2)
        shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d(layer2, weights['dec_w2'],strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        # print("decoder")
        layer3 = tf.nn.relu(layer3)
        # print(layer1.get_shape())
        # print(layer2.get_shape())
        # print(layer3.get_shape())
        return layer3
    
    def partial_fit(self, X, lr = 0.001): #  
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
        save_path = self.saver.save(self.sess,self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

# class Co
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
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
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
        print(latent.shape)
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
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))],max_to_keep = None)              
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


        return  layer3, pool1,  shapes #add olayer2 as 2nd arg
    
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
        # tf.train.Saver(max_to_keep=None)
        # self.saver(max_to_keep = None)
        save_path = self.saver.save(self.sess, savetmp )
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

class overConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, learning_rate = 0.001, batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = './mymodels/over/logs', no=0):
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
                   
        z = tf.reshape(latent, [batch_size, -1])  
        Coef = weights['Coef']         
        z_c = tf.matmul(Coef,z)    
        self.Coef = Coef        
        latent_c = tf.reshape(z_c, tf.shape(latent)) 
        self.z = z       
        
        self.x_r = self.decoder(latent_c, weights, shape)                
        
        print(self.x.shape,self.x_r.shape)
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
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0],  self.n_hidden[0],1],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[1],self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        # all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[2],self.n_hidden[1]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        
        
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
        
        # all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        # all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        # all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
        #####################################################
        # all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[0],self.n_hidden[1]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        # all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[1],self.n_hidden[1]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))
        #####################################################


        # all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
        #     initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        # all_weights['dec_b2'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
        
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        # print(shapes)
        shapes_en = shapes[0]
        # print
        # print(shapes_en)
        layer1 = tf.add(tf.nn.conv2d_transpose(x, weights['enc_w0'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*2,64,3]),\
            strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['enc_w1'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*4,shapes_en[2]*4,6]),\
         strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())

        layer2 = tf.layers.max_pooling2d(inputs=layer2, pool_size=[65,65], strides=8)

        # print(layer2.shape)
        # layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['enc_w2'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*4,shapes_en[2]*4,1]),\
        #  strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        # layer3 = tf.nn.relu(layer3)
        # # print("encoder")
        # shapes.append(layer3.get_shape().as_list())
        return  layer2, shapes
    
    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        # shape_de1 = shapes[3]
        # print(shapes)
        # print("decoder")
        # print(z.shape,weights['dec_w0'],weights['dec_b0'])
        shapes = [[None,32,32,1],[None,16,16,3],[None,8,8,6]]
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,1,1,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0],16,16,shape_de2[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)

        # shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0],32,32,1]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        print(layer1.shape,layer2.shape,layer3.shape)

        ##########################################
        # layer1 = tf.add(tf.nn.conv2d(z, weights['dec_w0'],strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        # layer1 = tf.nn.relu(layer1)
        # shape_de2 = shapes[1]
        # layer2 = tf.add(tf.nn.conv2d(layer1, weights['dec_w1'],strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        
        # layer2 = tf.nn.relu(layer2)
        ##########################################

        # shape_de3= shapes[0]
        # layer3 = tf.add(tf.nn.conv2d(layer2, weights['dec_w2'],strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        # # print("decoder")
        # layer3 = tf.nn.relu(layer3)
        # print(layer1.get_shape())
        # print(layer2.get_shape())
        # print(layer3.get_shape())
        return layer3
    
    def partial_fit(self, X, lr = 0.001): #  
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
        self.Saver(max_to_keep=None)
        save_path = self.saver.save(self.sess, savetmp)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

class orgConvAE20(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, reg = None, batch_size = 256,\
        denoise = False, model_path = None, logs_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL20CodeModel/new/pretrain/logs',no=0):  
    #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path        
        self.kernel_size = kernel_size      
        self.iter = 0
        self.batch_size = batch_size
        weights = self._initialize_weights()
        self.no = 0 
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        
        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)

        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))

            latent,shape = self.encoder(x_input, weights)
        self.z_conv = tf.reshape(latent,[batch_size, -1])       
        self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)   
        self.Coef = Coef                        
        latent_de_ft = tf.reshape(self.z_ssc, tf.shape(latent))     
        self.x_r_ft = self.decoder(latent_de_ft, weights, shape)        
                

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
            
        
        self.cost_ssc = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.z_conv,self.z_ssc), 2))
        self.recon_ssc =  tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
        self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef,2))
        tf.summary.scalar("ssc_loss", self.cost_ssc)
        tf.summary.scalar("reg_lose", self.reg_ssc)     
        
        self.loss_ssc = self.cost_ssc*reg_const2 + reg_const1*self.reg_ssc + self.recon_ssc

        self.merged_summary_op = tf.summary.merge_all()     
        self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
        

        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[0], self.kernel_size[0],1, n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
        return all_weights


    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        return  layer1, shapes

    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[0]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        
        return layer1

    def reconstruct(self,X):
        print(self.x_r_ft.shape)
        print(self.x.shape)
        return self.sess.run(self.x_r_ft, feed_dict = {self.x:X})

    def selfexpressive_moduel(self,batch_size):
        
        Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')          
        z_ssc = tf.matmul(Coef, self.z_conv)
        return z_ssc, Coef


    def finetune_fit(self, X, lr):
        
        C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
                                                    feed_dict = {self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return C, l1_cost,l2_cost 
    
    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)    

    def transform(self, X):
        return self.sess.run(self.z_conv, feed_dict = {self.x:X})

    def save_model(self):
        self.no = self.no+1
        savetmp = self.model_path + "%d.ckpt"%(self.no)
        # save_path = self.saver.save(self.sess,self.model_path)
        # tf.train.Saver(max_to_keep=None)
        # self.saver(max_to_keep = None)
        self.Saver(max_to_keep=None)
        save_path = self.saver.save(self.sess, savetmp )
        # print ("model saved in file: %s" % save_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.model_path)
        print ("model restored")

class newConvAE20(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, reg = None, batch_size = 256,\
        denoise = False, model_path = None, logs_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL20CodeModel/new/pretrain/logs',no=0):  
    #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path        
        self.kernel_size = kernel_size      
        self.iter = 0
        self.batch_size = batch_size
        weights = self._initialize_weights()
        self.no = 0 
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        
        if denoise == False:
            x_input = self.x
            latent, pool1, shape = self.encoder(x_input, weights)

        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))

            latent,shape = self.encoder(x_input, weights)

        latent = tf.add(latent,pool1)

        self.z_conv = tf.reshape(latent,[batch_size, -1])       
        self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)   
        self.Coef = Coef                        
        latent_de_ft = tf.reshape(self.z_ssc, tf.shape(latent))     
        self.x_r_ft = self.decoder(latent_de_ft, weights, shape)        
                

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))],max_to_keep =None) 
            
        
        self.cost_ssc = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.z_conv,self.z_ssc), 2))
        self.recon_ssc =  tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
        self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef,2))
        # self.adv_ssc = tf.reduce_sum(tf)
        tf.summary.scalar("ssc_loss", self.cost_ssc)
        tf.summary.scalar("reg_lose", self.reg_ssc)     
        
        self.loss_ssc = self.cost_ssc*reg_const2 + reg_const1*self.reg_ssc + self.recon_ssc

        self.merged_summary_op = tf.summary.merge_all()     
        self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        
        all_weights['oenc_w0'] = tf.get_variable("oenc_w0", shape=[self.kernel_size[0], self.kernel_size[0],  self.n_hidden[0],1],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['oenc_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))

        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))


        return all_weights


    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        shapes_en = shapes[0]

        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        # print("here")
        # print(shapes_en)
        olayer1 = tf.add(tf.nn.conv2d_transpose(x, weights['oenc_w0'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*2,64,self.n_hidden[0]]),\
            strides=[1,2,2,1],padding='SAME'),weights['oenc_b0'])
        olayer1 = tf.nn.relu(olayer1)
        pool1 = tf.layers.max_pooling2d(inputs=olayer1, pool_size=[2,2], strides=4)

        print(layer1.shape,pool1.shape)
        return  layer1, pool1,shapes

    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[0]
        # shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)

        # print(layer1.shape)
        return layer1

    def reconstruct(self,X):
        print(self.x_r_ft.shape)
        print(self.x.shape)
        return self.sess.run(self.x_r_ft, feed_dict = {self.x:X})

    def selfexpressive_moduel(self,batch_size):
        
        Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')          
        z_ssc = tf.matmul(Coef, self.z_conv)
        return z_ssc, Coef


    def finetune_fit(self, X, lr):
        C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
                                                    feed_dict = {self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return C, l1_cost,l2_cost 
    
    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)    

    def transform(self, X):
        return self.sess.run(self.z_conv, feed_dict = {self.x:X})

    def save_model(self):
        self.no = self.no+1
        savetmp = self.model_path + "%d.ckpt"%(self.no)
        # save_path = self.saver.save(self.sess,self.model_path)
        # tf.train.Saver(max_to_keep=None)
        # self.saver(max_to_keep = None)
        # self.saver(max_to_keep=None)
        save_path = self.saver.save(self.sess, savetmp )
        # print ("model saved in file: %s" % save_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.model_path)
        print ("model restored")


def ae_feature_clustering(CAE, X):
    CAE.restore()
    
    #eng = matlab.engine.start_matlab()
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/SSC_ADMM_v1.1',nargout=0)
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/EDSC_release',nargout=0)
    
    Z = CAE.transform(X)
    
    sio.savemat('AE_YaleB.mat', dict(Z = Z) )
    
    return

def train_face(Img, CAE, n_input, batch_size):    
    it = 0
    display_step = 500
    save_step = 2
    _index_in_epoch = 0
    _epochs= 0
    lr = 0.001
    # print("IN")
    # CAE.restore()
    # train the network
    while True:
        batch_x,  _index_in_epoch, _epochs =  next_batch(Img, _index_in_epoch , batch_size , _epochs)
        batch_x = np.reshape(batch_x,[batch_size,n_input[0],n_input[1],1])
        cost = CAE.finetune_fit(batch_x, lr)
        it = it +1
        # print(cost,batch_size)
        avg_cost = cost[0]/batch_size

        if it % display_step == 0:
            print ("epoch: %.1d" % _epochs)
            # print  ("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
            # break
            # test_face(Img, CAE, n_input)
        if it == 100000:
            break


    return

def test_face(Img, CAE, n_input):
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

if __name__ == '__main__':
    
    data = sio.loadmat('./Data/COIL20.mat')
    Img = data['fea']
    Label = data['gnd']
    Img = np.reshape(Img,(Img.shape[0],32,32,1))

    n_input = [32,32]
    kernel_size = [3]
    n_hidden = [15,6,6]
    batch_size = 20*72

    model_path = './mymodels/coil20/modelovern' 
    # restore_path = './pretrain-model-ORL/model-335-32x32-orl.ckpt' 
    logs_path = './ft/logs'

    restore_path = './pretrain-model-COIL20/model.ckpt'
    # # ft_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL20CodeModel/new/pretrain/model.ckpt'
    # logs_path = './ft/logs'

    num_class = 20 #how many class we sample
    num_sa = 72

    batch_size_test = num_sa * num_class


    iter_ft = 0
    ft_times = 50
    display_step = 5
    alpha = 0.04
    learning_rate = 1e-3

    reg1 = 1.0
    reg2 = 150.0

    # CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2, kernel_size = kernel_size, \
                # batch_size = batch_size_test, model_path = model_path, logs_path= logs_path)

    CAE = newConvAE20(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2, kernel_size = kernel_size, \
            batch_size = batch_size_test, model_path = model_path, logs_path= logs_path)


    train_face(Img, CAE, n_input, batch_size)
    #X = np.reshape(Img, [Img.shape[0],n_input[0],n_input[1],1])
    #ae_feature_clustering(CAE, X)
    
    