#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/25 14:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

initializer = tf.random_normal_initializer(0., 0.02)
activation_fn = tf.nn.elu

class BEGAN():
    def __init__(self, batch_size, data_size, filters, input_size, embedding, gamma):

        self.batch_size = batch_size
        self.data_size = data_size
        self.mm = 0.5
        self.mm2 = 0.999
        self.lamda = 1e-3
        self.gamma = gamma
        self.filter_number = filters
        self.input_size = input_size
        self.embedding = embedding

        self._build_model()

    def _build_model(self):
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_size, self.data_size, 3], name='y')
        self.kt = tf.placeholder(tf.float32, name='kt')
        self.lr = tf.placeholder(tf.float32, name='lr')

        # Generator
        self.recon_gen = self.generator(self.x)

        # Discriminator (Critic)
        d_real = self.decoder(self.encoder(self.y))
        d_fake = self.decoder(self.encoder(self.recon_gen))
        self.recon_dec = self.decoder(self.x)

        # Loss
        self.d_real_loss = self.l1_loss(self.y, d_real)
        self.d_fake_loss = self.l1_loss(self.recon_gen, d_fake)
        self.d_loss = self.d_real_loss - self.kt * self.d_fake_loss
        self.g_loss = self.d_fake_loss
        self.m_global = self.d_real_loss + tf.abs(self.gamma * self.d_real_loss - self.d_fake_loss)

        # Variables
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gen")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "disc_")
        for elem in g_vars:
            print(elem)
        for elem in d_vars:
            print(elem)

        # Optimizer
        self.opt_g = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.g_loss, var_list=g_vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.d_loss, var_list=d_vars)

    def generator_ops(self):
        return self.opt_g, self.g_loss, self.d_real_loss, self.d_fake_loss

    def discriminator_ops(self):
        return self.opt_d, self.d_loss

    def generator(self, x):
        with tf.variable_scope('gen', reuse=tf.AUTO_REUSE) as scope:

            w = self.data_size
            f = self.filter_number
            p = "SAME"

            x = tf.layers.dense(inputs=x, units=8 * 8 * f, activation=None, kernel_initializer=initializer)
            x = tf.reshape(tensor=x, shape=(-1, 8, 8, f))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w / 4), int(w / 4)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w / 2), int(w / 2)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w), int(w)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=3, kernel_size=(3, 3), padding='same', activation=None,
                                 kernel_initializer=initializer)
        return x

    def encoder(self, x):
        with tf.variable_scope('disc_encoder', reuse=tf.AUTO_REUSE) as scope:
            f = self.filter_number
            h = self.embedding

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=2 * f, kernel_size=(3, 3), padding='same', activation=None,
                                 kernel_initializer=initializer)
            x = tf.layers.average_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), padding='same')
            x = tf.layers.conv2d(inputs=x, filters=2 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=2 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=3 * f, kernel_size=(3, 3), padding='same', activation=None,
                                 kernel_initializer=initializer)
            x = tf.layers.average_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), padding='same')
            x = tf.layers.conv2d(inputs=x, filters=3 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=3 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=4 * f, kernel_size=(3, 3), padding='same', activation=None,
                                 kernel_initializer=initializer)
            x = tf.layers.average_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), padding='same')
            x = tf.layers.conv2d(inputs=x, filters=4 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=4 * f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.reshape(tensor=x, shape=(-1, 8 * 8 * 256))
            x = tf.layers.dense(inputs=x, units=h, activation=None, kernel_initializer=initializer)

        return x

    def decoder(self, x):
        with tf.variable_scope('disc_decoder', reuse=tf.AUTO_REUSE) as scope:

            w = self.data_size
            f = self.filter_number

            x = tf.layers.dense(inputs=x, units=8 * 8 * f, activation=None, kernel_initializer=initializer)
            x = tf.reshape(tensor=x, shape=(-1, 8, 8, f))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w / 4), int(w / 4)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w / 2), int(w / 2)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.image.resize_nearest_neighbor(images=x, size=(int(w), int(w)))

            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)
            x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=(3, 3), padding='same', activation=activation_fn,
                                 kernel_initializer=initializer)

            x = tf.layers.conv2d(inputs=x, filters=3, kernel_size=(3, 3), padding='same', activation=None,
                                 kernel_initializer=initializer)
        return x

    @staticmethod
    def l1_loss(x, y):
        return tf.reduce_mean(tf.abs(x - y))