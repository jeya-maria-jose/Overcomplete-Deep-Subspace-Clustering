
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
# from sklearn.manifold import TSNE
#import matlab.engine
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

class ODSC(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, learning_rate = 0.001,batch_size = 200, reg = None, \
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
        
        tf.summary.scalar("reg_loss", reg_const1 * self.reg_losses )
        
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        
        tf.summary.scalar("selfexpress_loss", reg_const2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost + reg_const1 * self.reg_losses + reg_const2 * self.selfexpress_losses  
        
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
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)

        shapes_en = shapes[0]

        olayer1 = tf.add(tf.nn.conv2d_transpose(x, weights['oenc_w0'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*2,56,self.n_hidden[0]]),\
            strides=[1,2,2,1],padding='SAME'),weights['oenc_b0'])
        olayer1 = tf.nn.relu(olayer1)
        # shapes.append(layer1.get_shape().as_list())
        olayer2 = tf.add(tf.nn.conv2d_transpose(olayer1, weights['oenc_w1'],tf.stack([tf.shape(self.x)[0],shapes_en[1]*4,shapes_en[2]*4,self.n_hidden[2]]),\
         strides=[1,2,2,1],padding='SAME'),weights['oenc_b1'])
        olayer2 = tf.nn.relu(olayer2)

        pool1 = tf.layers.max_pooling2d(inputs=olayer2, pool_size=[2,2], strides=32)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)

        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)

  
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
        return layer3

    
    
    def finetune_fit(self, X, lr): #  
        cost, summary, _, Coef = self.sess.run((self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef), feed_dict = {self.x: X, self.learning_rate: lr})#
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        # tr = TSNE(perplexity=50).fit_transform(self.z.eval(feed_dict = {self.x: X, self.learning_rate: lr}))
        # plt.scatter(tr[:, 0], tr[:, 1])
        # plt.show()

        return Coef,cost
    
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
    # print(c_x,gt_s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 
        
  
def test_facep(Img, CAE, n_input):
    # print(Img.shape)
    batch_x_test = Img
    # print(batch_x_test.shape)
    # batch_x_test= np.reshape(batch_x_test,[100,n_input[0],n_input[1],1])
    # print(batch_x_test.shape)
    CAE.restore()
    print(batch_x_test.shape)
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
    lr = 0.001
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
    data = sio.loadmat('.././Data/mnist-original.mat')
    
    print(data.keys())
    Img = data['data']
    Label = data['label']
    # print(Img.shape)
    # print(Label.shape)
    # Img = np.reshape(Img,(Img.shape[0],28,28,1))
    # print(Img.shape)
    Imgsub = np.zeros([784,1000])
    Labelsub = np.zeros([1000])
    sub_n = -1
    #Create MNIST sub dataset
    tmp = 1
    for class_n in range(0,10):
        for i in range(0,Label.shape[1]):
            class_num = int(Label[:,i])
            # print(class_num)
            # print(class_num,class_n)
            if class_num == class_n:
                sub_n = sub_n+1
                # print(sub_n, class_n)
                Imgsub[:,sub_n] = Img[:,i]
                Labelsub[sub_n] = Label[:,i] 
                tmp = 0
            if tmp == 0 and (sub_n+1) % 100 == 0 and sub_n!=0:
                tmp=1
                break

    # print(Imgsub.shape)
    # print(Labelsub)

    # exit()

    Img = Imgsub
    Label = Labelsub
    Img = np.transpose(Img)
    # print(Img.shape)
    # print(np.unique(Img[:,1]))
    Img = np.reshape(Img,(1000,28,28,1))
    # print(Img.shape)
    noise = np.random.rand(1000,28,28,1) * 0.5
    Img = Img + noise
    # Img = np.reshape(Img,(1000,28,28,1))
    n_input = [28,28]
    kernel_size = [5,3,3]
    n_hidden = [20,10,5]
    batch_size = 1000

    # model_path = './mymodels/mnist/model'
    # ft_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL100CodeModel/COIL100/pretrain/model50.ckpt'
    # logs_path = './ft/logs'

    _index_in_epoch = 0
    _epochs= 0
    num_class = 10 #how many class we sample
    # num_sa = 72
    batch_size_test = 1000


    iter_ft = 0
    ft_times = 20
    display_step = 1
    alpha = 0.092
    learning_rate = 0.001


    
    all_subjects = [10]
    
    avg = []
    med = []
    
    iter_loop = 0
    tot_ep = 100
    i = 50
    best_m = 0
    bestep = 0
    testep = 1
    reg1 = 20
    reg2 = 0.1
    logs_path = './ft/logs'
    for i in range(100,101):
        
        model_path = './pretrained/mnist.ckpt' 

        CAE = ODSC(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2, kernel_size = kernel_size, \
                    batch_size = batch_size_test, model_path = model_path, logs_path= logs_path,restore_path=model_path)

        acc_= []
        for j in range(0,1):
            coil20_all_subjs = Img
            coil20_all_subjs = coil20_all_subjs.astype(float)   
            label_all_subjs = Label
            label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
            label_all_subjs = np.squeeze(label_all_subjs) 
                
            CAE.initlization()
            CAE.restore()
            # test_facep(Img, CAE, Img)
            for iter_ft  in range(0,ft_times):
                iter_ft = iter_ft+1
                C,l2_cost = CAE.finetune_fit(coil20_all_subjs,learning_rate)
                if iter_ft % display_step == 0:
                    # print ("epoch: %.1d" % iter_ft, "cost: %.8f" % (l1_cost/float(batch_size_test)))
                    C = thrC(C,alpha)
                    
                    y_x, CKSym_x = post_proC(C, num_class, 12 ,8)
                    missrate_x = err_rate(label_all_subjs,y_x)
                                
                    acc = 1 - missrate_x
                    print ("experiment: %d" % iter_ft,"acc: %.4f" % acc)
                    m=acc
                    if best_m<m:
                        best_m = m
                        bestep = iter_ft
                    print("BEST error: %.4f%% at epoch %d" % (1-best_m,bestep))


            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
