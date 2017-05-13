import os,path,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time

class BatchGenerator:
    def __init__(self):
        self.folderPath = "sep"
        self.imagePath = glob.glob(self.folderPath+"/*/*.png")
        print "loaded %d images"%len(self.imagePath)
        self.imgSize = (64,64)
        assert self.imgSize[0]==self.imgSize[1]

    def getOnes(self,idx):
        x   = np.zeros( (len(idx),self.imgSize[0],self.imgSize[1],1), dtype=np.float32)

        for i in range(len(idx)):
            img = cv2.imread(self.imagePath[idx[i]],cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img,axis=2)
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            img = cv2.resize(img,self.imgSize)
            img = np.expand_dims(img,axis=2)
            x[i,:,:,:] = img / 255.

        return x,None

    def getOne(self,idx):
        x   = np.zeros( (1,self.imgSize[0],self.imgSize[1],1), dtype=np.float32)

        img = cv2.imread(self.imagePath[idx],cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img,axis=2)
        dmin = min(img.shape[0],img.shape[1])
        img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
        img = cv2.resize(img,self.imgSize)
        img = np.expand_dims(img,axis=2)
        x[0,:,:,:] = img / 255.

        return x,None

    def getBatch(self,nBatch):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],1), dtype=np.float32)
        for i in range(nBatch):
            img = cv2.imread(self.imagePath[random.randint(0,len(self.imagePath)-1)],cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img,axis=2)
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            img = cv2.resize(img,self.imgSize)
            img = np.expand_dims(img,axis=2)
            x[i,:,:,:] = img / 255.

        return x,None

class VAEGAN:
    def __init__(self,isTraining,imageSize,labelSize,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.labelSize = labelSize
        self.initOP = None
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildGenerator(self,z,label=None,reuse=False,isTraining=True):
        dim_0_h,dim_0_w = self.imageSize[0],self.imageSize[1]
        dim_1_h,dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h,dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h,dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)
        dim_4_h,dim_4_w = self.calcImageSize(dim_3_h, dim_3_w, stride=2)

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            if label:
                l = tf.one_hot(label,self.labelSize,name="label_onehot")
                h = tf.concat([z,l],axis=1,name="concat_z")
                dim_next = self.zdim + self.labelSize
            else:
                h = z
                dim_next = self.zdim

            # fc1
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([dim_next,dim_4_h*dim_4_w*128],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b
            h = self.leakyReLU(h)
            #
            h = tf.reshape(h,(self.nBatch,dim_4_h,dim_4_h,128))

            # deconv4
            self.d_deconv4_w, self.d_deconv4_b = self._deconv_variable([5,5,128,64],name="deconv4")
            h = self._deconv2d(h,self.d_deconv4_w, output_shape=[self.nBatch,dim_3_h,dim_3_w,64], stride=2) + self.d_deconv4_b
            h = self.leakyReLU(h)

            # deconv3
            self.d_deconv3_w, self.d_deconv3_b = self._deconv_variable([5,5,64,32],name="deconv3")
            h = self._deconv2d(h,self.d_deconv3_w, output_shape=[self.nBatch,dim_2_h,dim_2_w,32], stride=2) + self.d_deconv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm1")
            h = self.leakyReLU(h)

            # deconv2
            self.d_deconv2_w, self.d_deconv2_b = self._deconv_variable([5,5,32,8],name="deconv2")
            h = self._deconv2d(h,self.d_deconv2_w, output_shape=[self.nBatch,dim_1_h,dim_1_w,8], stride=2) + self.d_deconv2_b
            h = self.leakyReLU(h)

            # deconv1
            self.d_deconv1_w, self.d_deconv1_b = self._deconv_variable([5,5,8,1],name="deconv1")
            h = self._deconv2d(h,self.d_deconv1_w, output_shape=[self.nBatch,dim_0_h,dim_0_w,1], stride=2) + self.d_deconv1_b
            self.h = self.d_deconv1_b

            # sigmoid
            y = tf.sigmoid(h)

            ### summary
            if reuse:
                tf.summary.histogram("d_fc1_w"   ,self.d_fc1_w)
                tf.summary.histogram("d_fc1_b"   ,self.d_fc1_b)
                tf.summary.histogram("d_deconv1_w"   ,self.d_deconv1_w)
                tf.summary.histogram("d_deconv1_b"   ,self.d_deconv1_b)
                tf.summary.histogram("d_deconv2_w"   ,self.d_deconv2_w)
                tf.summary.histogram("d_deconv2_b"   ,self.d_deconv2_b)
                tf.summary.histogram("d_deconv3_w"   ,self.d_deconv3_w)
                tf.summary.histogram("d_deconv3_b"   ,self.d_deconv3_b)
                tf.summary.histogram("d_deconv4_w"   ,self.d_deconv4_w)
                tf.summary.histogram("d_deconv4_b"   ,self.d_deconv4_b)

        return y

    def buildEncoder(self,y,isTraining=True,label=None,reuse=False):
        with tf.variable_scope("Encoder") as scope:
            if reuse: scope.reuse_variables()

            # conditional layer
            if label:
                l = tf.one_hot(label,self.labelSize,name="label_onehot")
                l = tf.reshape(l,[self.nBatch,1,1,self.labelSize])
                k = tf.ones([self.nBatch,self.imageSize[0],self.imageSize[1],self.labelSize])
                k = k * l
                h = tf.concat([y,k],axis=3)
                dim_next = 1+self.labelSize
            else:
                h = y
                dim_next = 1

            # conv1
            self.e_conv1_w, self.e_conv1_b = self._conv_variable([5,5,dim_next,8],name="econv1")
            h = self._conv2d(h,self.e_conv1_w, stride=2) + self.e_conv1_b
            h = self.leakyReLU(h)

            # conv2
            self.e_conv2_w, self.e_conv2_b = self._conv_variable([5,5,8,32],name="econv2")
            h = self._conv2d(h,self.e_conv2_w, stride=2) + self.e_conv2_b
            h = self.leakyReLU(h)

            # conv3
            self.e_conv3_w, self.e_conv3_b = self._conv_variable([5,5,32,64],name="econv3")
            h = self._conv2d(h,self.e_conv3_w, stride=2) + self.e_conv3_b
            h = self.leakyReLU(h)
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="eNorm1")

            # conv4
            self.e_conv4_w, self.e_conv4_b = self._conv_variable([5,5,64,128],name="econv4")
            h = self._conv2d(h,self.e_conv4_w, stride=2) + self.e_conv4_b
            h = self.leakyReLU(h)

            h_mu = h_sigma = h

            # fc_mu
            n_b, n_h, n_w, n_f = [int(x) for x in h_mu.get_shape()]
            h_mu = tf.reshape(h_mu,[self.nBatch,n_h*n_w*n_f])
            self.e_fc_mu_w, self.e_fc_mu_b = self._fc_variable([n_h*n_w*n_f,self.zdim],name="fc_mu")
            h_mu = tf.matmul(h_mu, self.e_fc_mu_w) + self.e_fc_mu_b

            # fc_sigma
            n_b, n_h, n_w, n_f = [int(x) for x in h_sigma.get_shape()]
            h_sigma = tf.reshape(h_sigma,[self.nBatch,n_h*n_w*n_f])
            self.e_fc_sigma_w, self.e_fc_sigma_b = self._fc_variable([n_h*n_w*n_f,self.zdim],name="fc_sigma")
            h_lnsigma = tf.matmul(h_sigma, self.e_fc_sigma_w) + self.e_fc_sigma_b

            ### summary
            if not reuse:
                tf.summary.histogram("e_conv1_w"   ,self.e_conv1_w)
                tf.summary.histogram("e_conv1_b"   ,self.e_conv1_b)
                tf.summary.histogram("e_conv2_w"   ,self.e_conv2_w)
                tf.summary.histogram("e_conv2_b"   ,self.e_conv2_b)
                tf.summary.histogram("e_conv3_w"   ,self.e_conv3_w)
                tf.summary.histogram("e_conv3_b"   ,self.e_conv3_b)
                tf.summary.histogram("e_conv4_w"   ,self.e_conv4_w)
                tf.summary.histogram("e_conv4_b"   ,self.e_conv4_b)
                tf.summary.histogram("e_fc_mu_w"   ,self.e_fc_mu_w)
                tf.summary.histogram("e_fc_mu_b"   ,self.e_fc_mu_b)
                tf.summary.histogram("e_fc_sigma_w"   ,self.e_fc_sigma_w)
                tf.summary.histogram("e_fc_sigma_b"   ,self.e_fc_sigma_b)

        return h_mu,h_lnsigma

    def buildDiscriminator(self,y,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()

            h = y
            # conv1
            self.d_conv1_w, self.d_conv1_b = self._conv_variable([5,5,1,64],name="conv1")
            h = self._conv2d(h,self.d_conv1_w, stride=2) + self.d_conv1_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.d_conv2_w, self.d_conv2_b = self._conv_variable([5,5,64,128],name="conv2")
            h = self._conv2d(h,self.d_conv2_w, stride=2) + self.d_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            # conv3
            self.d_conv3_w, self.d_conv3_b = self._conv_variable([5,5,128,256],name="conv3")
            h = self._conv2d(h,self.d_conv3_w, stride=2) + self.d_conv3_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm3")
            h = self.leakyReLU(h)

            # conv4
            self.d_conv4_w, self.d_conv4_b = self._conv_variable([5,5,256,512],name="conv4")
            h = self._conv2d(h,self.d_conv4_w, stride=2) + self.d_conv4_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm4")
            h = self.leakyReLU(h)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
            h = tf.reshape(h,[self.nBatch,n_h*n_w*n_f])
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([n_h*n_w*n_f,1],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b

            ### summary
            if not reuse:
                tf.summary.histogram("d_fc1_w"   ,self.d_fc1_w)
                tf.summary.histogram("d_fc1_b"   ,self.d_fc1_b)
                tf.summary.histogram("d_conv1_w"   ,self.d_conv1_w)
                tf.summary.histogram("d_conv1_b"   ,self.d_conv1_b)
                tf.summary.histogram("d_conv2_w"   ,self.d_conv2_w)
                tf.summary.histogram("d_conv2_b"   ,self.d_conv2_b)
                tf.summary.histogram("d_conv3_w"   ,self.d_conv3_w)
                tf.summary.histogram("d_conv3_b"   ,self.d_conv3_b)
                tf.summary.histogram("d_conv4_w"   ,self.d_conv4_w)
                tf.summary.histogram("d_conv4_b"   ,self.d_conv4_b)

        return h

    def buildModel(self):
        # define variables
        self.x        = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 1],name="inputImage")
        self.y_real   = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 1],name="realImage")

        self.z_mu, self.z_lnsigma = self.buildEncoder(self.x,isTraining=self.isTraining) # lnsigma = ln(sigma)... This admits to take -inf,+inf and make the calculation easier
        # z -> [u_1,u_2,...],[s_1,s_2,...]
        rand        = tf.random_normal([self.nBatch,self.zdim]) # normal distribution
        self.z      = rand * tf.exp(self.z_lnsigma) + self.z_mu

        self.y_fake = self.buildGenerator(self.z,isTraining=self.isTraining)

        self.d_real  = self.buildDiscriminator(self.y_real)
        self.d_fake  = self.buildDiscriminator(self.y_fake,reuse=True)

        self.z_sample = tf.placeholder(tf.float32, [self.nBatch, self.zdim],name="z_sample")
        self.y_sample = self.buildGenerator(self.z_sample,reuse=True,isTraining=False)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real,labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.zeros_like(self.d_fake)))
        self.g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.ones_like (self.d_fake)))

        #self.pix_loss = - tf.reduce_mean( self.x * tf.log( tf.clip_by_value(self.y_fake,1e-20,1e+20)) + (1.-self.x) * tf.log( tf.clip_by_value(1.-self.y_fake,1e-20,1e+20))) # bbernoulli negative log likelihood. May be replaced by RMS?
        self.pix_loss = tf.reduce_mean( tf.abs( self.x - self.y_fake ) ) # L1 loss is to make the image more clearer

        self.kl_loss = 0.5 * tf.reduce_mean(tf.square(self.z_mu) + tf.exp(self.z_lnsigma)**2 - 2.*self.z_lnsigma - 1.)

        self.e_loss      = self.kl_loss     + self.pix_loss
        self.g_loss      = self.g_loss_fake + self.pix_loss
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        # define optimizer
        #self.e_optimizer = tf.train.AdamOptimizer(self.learnRate,beta1=0.5).minimize(self.e_loss, var_list=[k for k in tf.trainable_variables() if "Encoder"       in k.name])
        self.e_optimizer = tf.train.AdamOptimizer(self.learnRate    ,beta1=0.5).minimize(self.e_loss, var_list=[k for k in tf.trainable_variables() if ("Encoder" in k.name) or ("Generator" in k.name)])
        self.g_optimizer = tf.train.AdamOptimizer(self.learnRate    ,beta1=0.5).minimize(self.g_loss, var_list=[k for k in tf.trainable_variables() if "Generator"     in k.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learnRate*0.5,beta1=0.5).minimize(self.d_loss, var_list=[k for k in tf.trainable_variables() if "Discriminator" in k.name])

        ### summary
        tf.summary.scalar("d_loss"      ,self.d_loss)
        tf.summary.scalar("g_loss"      ,self.g_loss)
        tf.summary.scalar("e_loss"      ,self.e_loss)
        tf.summary.scalar("pix_loss"    ,self.pix_loss)
        tf.summary.scalar("kl_loss"     ,self.kl_loss)
        tf.summary.scalar("d_loss_fake" ,self.d_loss_fake)
        tf.summary.scalar("d_loss_real" ,self.d_loss_real)
        tf.summary.scalar("g_loss_fake" ,self.g_loss_fake)
        tf.summary.histogram("z_mu"     ,self.z_mu   )
        tf.summary.histogram("z_sigma"  ,tf.exp(self.z_lnsigma))

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver(max_to_keep=0)
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        #############################
        ### initializer
        self.initOP = tf.global_variables_initializer()
        self.sess.run(self.initOP)

        return

    def train(self,f_batch):

        def tileImage(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d,w*d,3),dtype=np.float32)
            for idx,img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
            return r
        
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"images")):
            os.makedirs(os.path.join(self.saveFolder,"images"))

        self.loadModel(self.reload)

        step = -1
        start = time.time()
        while True:
            step += 1

            batch_images,_ = f_batch(self.nBatch)
            _,e_loss,kl_loss,pix_loss        = self.sess.run([self.e_optimizer,self.e_loss,self.kl_loss,self.pix_loss],             feed_dict={self.x:batch_images})
            _,g_loss,g_loss_fake,z,y_fake    = self.sess.run([self.g_optimizer,self.g_loss,self.g_loss_fake,self.z_mu,self.y_fake], feed_dict={self.x:batch_images})
            _,d_loss,d_loss_real,d_loss_fake = self.sess.run([self.d_optimizer,self.d_loss,self.d_loss_real,self.d_loss_fake],      feed_dict={self.x:batch_images,self.y_real:batch_images})

            if step>0 and step%10==0:
                summary = self.sess.run(self.summary,feed_dict={self.x:batch_images,self.y_real:batch_images})
                self.writer.add_summary(summary,step)

            if step%500==0:
                #print "%6d:  time/step = %.2f sec"%(step, time.time()-start)
                print "%6d: loss(e) = loss(KL)+loss(pix) = %.4e + %.4e = %.4e, loss(g) = loss(fake)+loss(pix) = %.4e + %.4e = %.4e, loss(e) = loss(real)+loss(fake) = %.4e + %.4e = %.4e  time/step = %.2f sec"%(step,kl_loss,pix_loss,e_loss, g_loss_fake,pix_loss,g_loss, d_loss_real,d_loss_fake,d_loss, time.time()-start)
                start = time.time()

                #l0 = np.array([x%10 for x in range(self.nBatch)],dtype=np.int32)
                z = np.random.normal(0.,1.,[self.nBatch,self.zdim])
                #z1 = np.zeros([self.nBatch,self.zdim])

                g_image = self.sess.run(self.y_sample,feed_dict={self.z_sample:z})
                #g_image2 = self.sess.run(self.y_sample,feed_dict={self.z:z2,self.l:l0})
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_real.png"%step),tileImage(batch_images)*255.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_reco.png"%step),tileImage(y_fake)*255.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_rand.png"%step),tileImage(g_image)*255.)
                #cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_fake2.png"%step),tileImage(g_image2)*255.)
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)

    def test(self,oriImage=None,title=None):
        def tileImage(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d,w*d,3),dtype=np.float32)
            for idx,img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
            return r
        
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"timages")):
            os.makedirs(os.path.join(self.saveFolder,"timages"))

        self.loadModel(self.reload)

        if not type(oriImage)==type(None):
            #print oriImage.shape
            z_ori = self.sess.run(self.z_mu,feed_dict={self.x:np.tile(oriImage,(self.nBatch,1,1,1))})[0]
        #print z_ori
        z = np.zeros((self.nBatch,self.zdim))
        if not type(oriImage)==type(None):
            z1 = np.random.normal(0,1.0,[self.zdim])
            #z1 = np.zeros((self.zdim))
            for i in range(self.nBatch):
                z[i,:] = (z_ori-z1)*float(i)/self.nBatch + (z_ori+z1)*(1-float(i)/self.nBatch)
        else:
            z1 = np.random.normal(0,1,[self.zdim])
            z2 = np.random.normal(0,1,[self.zdim])
            for i in range(self.nBatch):
                z[i,:] = z1*float(i)/self.nBatch + z2*(1-float(i)/self.nBatch)
        #z1 = np.tile(z1,(self.nBatch,1))
        #z2 = np.tile(z2,(self.nBatch,1))
        #kk = np.linspace(0,1,self.nBatch)
        #z  = z1*kk + z2[,:]*(1.-kk)
        g_image = self.sess.run(self.y_sample,feed_dict={self.z_sample:z})
        if title:
            cv2.imwrite(os.path.join(self.saveFolder,"timages","img_%s_gene.png"%title),tileImage(g_image)*255.)
