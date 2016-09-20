"""
This implements the idea of `Variational Neural Discourse Relation Recognizer`

Biao Zhang -- <zb@stu.xmu.edu.cn or biaojiaxing@gmail.com>
"""

import cPickle
import time
import sys

import numpy as np
import theano
import theano.tensor as T

"""
This code borrows some ideas from "https://github.com/y0ast/Variational-Autoencoder"
"""
class VarNDrr:
    def __init__(self,
            L_encoder,      # hidden for encoder dimension for label
            L_decoder,      # hidden for decoder dimension for label
            H_arg1_encoder, # hidden for encoder dimension for argument one
            H_arg1_decoder, # hidden for decoder dimension for argument one
            H_arg2_encoder, # hidden for encoder dimension for argument two
            H_arg2_decoder, # hidden for decoder dimension for argument two
            X_arg1,         # input dimension for argument one of discourse
            X_arg2,         # input dimension for argument two of discourse
            Z,              # variation dimension for Z
            batch_size,     # batch size training
            label_size,     # number of given labels
            L = 1,          # you know, the sample size for expection approximation
            learning_rate = 0.01, # learning rate for optimization
            clip_c=-1.):    # gradient clip
        # variables for argument 1
        self.H_arg1_encoder = H_arg1_encoder
        self.H_arg1_decoder = H_arg1_decoder
        self.X_arg1 = X_arg1

        # variable for argument 2
        self.H_arg2_encoder = H_arg2_encoder
        self.H_arg2_decoder = H_arg2_decoder
        self.X_arg2 = X_arg2

        assert X_arg1 == X_arg2, 'Sorry, the input dimension of arg1 and arg2 must to be equal ' +\
                'due to the parameter tying trick'

        self.Z = Z
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_size = label_size

        self.L_encoder = L_encoder
        self.L_decoder = L_decoder

        self.sigmaInit = 0.01
        self.lowerbound = 0
        self.clip_c = clip_c

    def initParams(self):
    	"""
        initialize all Ws and bs using gaussian distribution from numpy,
        these CPU variables is then converted into GPU variables via theano
        """
        # layer one
        Arg1_W1 = np.random.normal(0,self.sigmaInit,(self.H_arg1_encoder,self.X_arg1)).astype('float32')
        Arg1_b1 = np.zeros((self.H_arg1_encoder,)).astype('float32')

        # for inference
        Arg1_W2 = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg1_encoder)).astype('float32')
        Arg2_W2 = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg2_encoder)).astype('float32')
        b2 = np.zeros((self.Z,)).astype('float32')

        Arg1_W2_prior = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg1_encoder)).astype('float32')
        Arg2_W2_prior = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg2_encoder)).astype('float32')
        b2_prior = np.zeros((self.Z,)).astype('float32')

        Arg1_W3 = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg1_encoder)).astype('float32')
        Arg2_W3 = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg2_encoder)).astype('float32')
        b3 = np.zeros((self.Z,)).astype('float32')

        Arg1_W3_prior = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg1_encoder)).astype('float32')
        Arg2_W3_prior = np.random.normal(0,self.sigmaInit,(self.Z,self.H_arg2_encoder)).astype('float32')
        b3_prior = np.zeros((self.Z,)).astype('float32')

        # generation
        Label_W1 = np.random.normal(0,self.sigmaInit,(self.L_encoder,self.label_size)).astype('float32')
        Label_b1 = np.zeros((self.L_encoder,)).astype('float32')
        Label_W2 = np.random.normal(0,self.sigmaInit,(self.Z,self.L_encoder)).astype('float32')
        Label_W3 = np.random.normal(0,self.sigmaInit,(self.Z,self.L_encoder)).astype('float32')
        L_W4 = np.random.normal(0,self.sigmaInit,(self.L_decoder,self.Z)).astype('float32')
        L_b4 = np.zeros((self.L_decoder,)).astype('float32')

        L_Wc = np.random.normal(0,self.sigmaInit,(self.label_size,self.L_decoder)).astype('float32')
        L_bc = np.zeros((self.label_size,)).astype('float32')

        L_W6 = np.random.normal(0,self.sigmaInit,(self.L_decoder,self.L_decoder)).astype('float32')
        L_b6 = np.zeros((self.L_decoder,)).astype('float32')
        L_W7 = np.random.normal(0,self.sigmaInit,(self.L_decoder,self.L_decoder)).astype('float32')
        L_b7 = np.zeros((self.L_decoder,)).astype('float32')
        L_W8 = np.random.normal(0,self.sigmaInit,(self.L_decoder,self.L_decoder)).astype('float32')
        L_b8 = np.zeros((self.L_decoder,)).astype('float32')

        Arg1_W4 = np.random.normal(0,self.sigmaInit,(self.H_arg1_decoder,self.Z)).astype('float32')
        Arg1_b4 = np.zeros((self.H_arg1_decoder,)).astype('float32')
        Arg2_W4 = np.random.normal(0,self.sigmaInit,(self.H_arg2_decoder,self.Z)).astype('float32')
        Arg2_b4 = np.zeros((self.H_arg2_decoder,)).astype('float32')

        Arg1_W5 = np.random.normal(0,self.sigmaInit,(self.X_arg1,self.H_arg1_decoder)).astype('float32')
        Arg1_b5 = np.zeros((self.X_arg1,)).astype('float32')

        self.params = [\
            Arg1_W1, Arg1_W2, Arg1_W3, Arg1_W4, Arg1_W5, \
            Arg2_W2, Arg2_W3, Arg2_W4, \
            Arg1_W2_prior, Arg1_W3_prior, Arg2_W2_prior, Arg2_W3_prior, \
            Arg1_b1, Arg1_b4, Arg1_b5, Arg2_b4, b2, b3, \
            L_Wc, L_bc, Label_W1, Label_W2, Label_W3, \
            L_W4, Label_b1, L_b4, b2_prior, b3_prior, L_W6, L_b6, L_W7, L_b7, L_W8, L_b8]
	self.params_names = [\
            "Arg1_W1", "Arg1_W2", "Arg1_W3", "Arg1_W4", "Arg1_W5", \
            "Arg2_W2", "Arg2_W3", "Arg2_W4", \
            "Arg1_W2_prior", "Arg1_W3_prior", "Arg2_W2_prior", "Arg2_W3_prior", \
            "Arg1_b1", "Arg1_b4", "Arg1_b5", "Arg2_b4", "b2", "b3", \
            "L_Wc", "L_bc", "Label_W1", "Label_W2", "Label_W3", \
            "L_W4", "Label_b1", "L_b4", "b2_prior", "b3_prior", "L_W6", "L_b6", "L_W7", "L_b7", "L_W8", "L_b8"]

        self.tparams = [theano.shared(p, name=n) for p,n in zip(self.params, self.params_names)]

    def createGradientFunctions(self):
        # Create the Theano variables
        x_arg1 = T.matrix('x_arg1', dtype='float32')
        x_arg2 = T.matrix('x_arg2', dtype='float32')
        true_class = T.matrix('true_class', dtype='float32')
        eps = T.matrix("eps", dtype='float32')
        Arg1_W1, Arg1_W2, Arg1_W3, Arg1_W4, Arg1_W5, \
        Arg2_W2, Arg2_W3, Arg2_W4, \
        Arg1_W2_prior, Arg1_W3_prior, Arg2_W2_prior, Arg2_W3_prior, \
        Arg1_b1, Arg1_b4, Arg1_b5, Arg2_b4, b2, b3, \
        L_Wc, L_bc, Label_W1, Label_W2, Label_W3, \
        L_W4, Label_b1, L_b4, b2_prior, b3_prior, L_W6, L_b6, L_W7, L_b7, L_W8, L_b8 = self.tparams

        # Parameter Tying
        Arg2_W1 = Arg1_W1
        Arg2_b1 = Arg1_b1
        Arg2_W5 = Arg1_W5
        Arg2_b5 = Arg1_b5

        # Neural Inferencer
        h_arg1_encoder = T.tanh(T.dot(Arg1_W1,x_arg1) + Arg1_b1.dimshuffle(0, 'x'))
        h_arg2_encoder = T.tanh(T.dot(Arg2_W1,x_arg2) + Arg2_b1.dimshuffle(0, 'x'))
        l_encoder = T.tanh(T.dot(Label_W1,true_class) + Label_b1.dimshuffle(0, 'x'))

        mu_poster_encoder = T.dot(Arg1_W2,h_arg1_encoder) + T.dot(Arg2_W2,h_arg2_encoder) \
                + T.dot(Label_W2,l_encoder) + b2.dimshuffle(0, 'x')
        log_sigma_poster_encoder = \
                np.float32(0.5)*(T.dot(Arg1_W3,h_arg1_encoder) + T.dot(Arg2_W3,h_arg2_encoder) \
                + T.dot(Label_W3,l_encoder) + b3.dimshuffle(0, 'x'))

        mu_prior_encoder = T.dot(Arg1_W2_prior,h_arg1_encoder) + T.dot(Arg2_W2_prior,h_arg2_encoder) \
                + b2_prior.dimshuffle(0, 'x')
        log_sigma_prior_encoder = \
                np.float32(0.5)*(T.dot(Arg1_W3_prior,h_arg1_encoder) + T.dot(Arg2_W3_prior,h_arg2_encoder) \
                + b3_prior.dimshuffle(0, 'x'))

        #Find the hidden variable z
        z = mu_poster_encoder + T.exp(log_sigma_poster_encoder)*eps

        prior = T.sum((log_sigma_prior_encoder - log_sigma_poster_encoder) + \
                (T.exp(log_sigma_poster_encoder)**np.float32(2) + \
                (mu_poster_encoder - mu_prior_encoder)**np.float32(2)) /
                (np.float32(2)*(T.exp(log_sigma_prior_encoder)**np.float32(2))) - np.float32(0.5))

        #Neural Generator
        h_arg1_decoder = T.tanh(T.dot(Arg1_W4,z) + Arg1_b4.dimshuffle(0, 'x'))
        h_arg2_decoder = T.tanh(T.dot(Arg2_W4,z) + Arg2_b4.dimshuffle(0, 'x'))
        y_arg1 = T.nnet.sigmoid(T.dot(Arg1_W5,h_arg1_decoder) + Arg1_b5.dimshuffle(0, 'x'))
        y_arg2 = T.nnet.sigmoid(T.dot(Arg2_W5,h_arg2_decoder) + Arg2_b5.dimshuffle(0, 'x'))
        logpxz = -(T.nnet.binary_crossentropy(y_arg1,x_arg1).sum() \
                + T.nnet.binary_crossentropy(y_arg2,x_arg2).sum())

        l_decoder = T.tanh(T.dot(L_W4,z) + L_b4.dimshuffle(0, 'x'))
        l_pred_decoder = T.tanh(T.dot(L_W4, mu_prior_encoder) + L_b4.dimshuffle(0, 'x'))

        l_decoder = T.tanh(T.dot(L_W6,l_decoder) + L_b6.dimshuffle(0, 'x'))
        l_pred_decoder = T.tanh(T.dot(L_W6,l_pred_decoder) + L_b6.dimshuffle(0, 'x'))
        l_decoder = T.tanh(T.dot(L_W7,l_decoder) + L_b7.dimshuffle(0, 'x'))
        l_pred_decoder = T.tanh(T.dot(L_W7,l_pred_decoder) + L_b7.dimshuffle(0, 'x'))
        l_decoder = T.tanh(T.dot(L_W8,l_decoder) + L_b8.dimshuffle(0, 'x'))
        l_pred_decoder = T.tanh(T.dot(L_W8,l_pred_decoder) + L_b8.dimshuffle(0, 'x'))

        pred_class = T.nnet.softmax(T.dot(L_Wc,l_decoder) + L_bc.dimshuffle(0, 'x'))
        logpc = -(T.nnet.categorical_crossentropy(pred_class,true_class).sum())
        pred_level = T.nnet.softmax(T.dot(L_Wc,l_pred_decoder) + L_bc.dimshuffle(0, 'x'))

        logp = - logpxz - logpc + prior

        #Compute all the gradients
        derivatives = T.grad(logp,wrt=self.tparams)

        # apply gradient clipping here
        if self.clip_c > 0.:
            g2 = 0.
            for g in derivatives:
                g2 += (g**2).sum()
            new_grads = []
            for g in derivatives:
                new_grads.append(T.switch(g2 > (self.clip_c**2),
                                           g / T.sqrt(g2) * self.clip_c,
                                           g))
            derivatives = new_grads

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        self.gradientfunction = theano.function([x_arg1,x_arg2,true_class,eps], \
                derivatives, on_unused_input='ignore')
        self.lowerboundfunction = theano.function([x_arg1,x_arg2,true_class,eps], \
                logp, on_unused_input='ignore')
        self.predictionfunction = theano.function([x_arg1,x_arg2], \
                pred_level.T, on_unused_input='ignore')

        #Adam Optimizer
        # This code is adapted from https://github.com/nyu-dl/dl4mt-tutorial/blob/master/session2/nmt.py
        def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):
            gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in zip(self.params_names, tparams)]
            gsup = [(gs, g) for gs, g in zip(gshared, grads)]

            f_grad_shared = theano.function(inp, cost, updates=gsup, profile=False)

            updates = []

            t_prev = theano.shared(np.float32(0.))
            t = t_prev + 1.
            lr_t = lr * T.sqrt(1. - beta2**t) / (1. - beta1**t)

            for p, g in zip(tparams, gshared):
                m = theano.shared(p.get_value() * 0., p.name + '_mean')
                v = theano.shared(p.get_value() * 0., p.name + '_variance')
                m_t = beta1 * m + (1. - beta1) * g
                v_t = beta2 * v + (1. - beta2) * g**2
                step = lr_t * m_t / (T.sqrt(v_t) + e)
                p_t = p - step
                updates.append((m, m_t))
                updates.append((v, v_t))
                updates.append((p, p_t))
            updates.append((t_prev, t))

            f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=False)

            return f_grad_shared, f_update

        lr = T.scalar(name='lr')
        self.f_grad_shared, self.f_update = \
                adam(lr, self.tparams, derivatives[:-1], [x_arg1,x_arg2,true_class,eps], logp)

    def fillBatch(self, data, dataIdx, dim):
        N = len(dataIdx)
        miniBatch = np.zeros((N, dim), dtype='float32')

        for i in xrange(N):
            _miniBatch = np.asarray(data[dataIdx[i]])
            _miniBatch[_miniBatch>=dim] = 0  #We make the OOV index to be 0
                                             #So, the data in training and test file should be index from 1
            miniBatch[i][_miniBatch] = 1.0

        return miniBatch

    def getBatch(self, N, batch_size, shuffle=True):
        """Used to shuffle the dataset at each iteration."""
        idxList = np.arange(N, dtype="int64")
        if shuffle:
            np.random.shuffle(idxList)

        miniBatches = []
        miniBatchStart = 0
        for i in range(N // batch_size):
            miniBatches.append(idxList[miniBatchStart:miniBatchStart + batch_size])
            miniBatchStart += batch_size

        if miniBatchStart != N:
            # Make a minibatch out of what is left
            miniBatches.append(idxList[miniBatchStart:])
        return miniBatches

    def getPrediction(self,data_arg1,data_arg2,data_lbl):
       	"""Main method, slices data in minibatches and performs prediction"""
        N_arg1 = len(data_arg1)
        N_arg2 = len(data_arg2)
        N_lbl = len(data_lbl)
        assert N_arg1 == N_arg2 and N_arg1 == N_lbl, 'Hi~man! The arguments should be equal!'
        N = N_arg1

        batch_size = 128
        batches = self.getBatch(N, batch_size, shuffle=False)
        labels = []
        for i in xrange(0,len(batches)):
            miniBatch_arg1 = self.fillBatch(data_arg1,batches[i],self.X_arg1)
            miniBatch_arg2 = self.fillBatch(data_arg2,batches[i],self.X_arg2)
            miniBatch_lbl = self.fillBatch(data_lbl,batches[i],self.label_size)

            predictions = self.predictionfunction(x_arg1=miniBatch_arg1.T,x_arg2=miniBatch_arg2.T)

            for dIdx in xrange(miniBatch_lbl.shape[0]):
                pred_lbl = predictions[dIdx].argmax()
                true_lbl = miniBatch_lbl[dIdx].argmax()
                labels.append((pred_lbl, true_lbl))
        return labels

    def iterate(self,data_arg1,data_arg2,data_lbl):
       	"""Main method, slices data in minibatches and performs an iteration"""
        N_arg1 = len(data_arg1)
        N_arg2 = len(data_arg2)
        N_lbl = len(data_lbl)
        assert N_arg1 == N_arg2 and N_arg1 == N_lbl, 'Hi~man! The arguments should be equal!'
        N = N_arg1

        batches = self.getBatch(N, self.batch_size)
        for i in xrange(0,len(batches)):
            print 'batches training %d'%(i+1)
            miniBatch_arg1 = self.fillBatch(data_arg1,batches[i],self.X_arg1)
            miniBatch_arg2 = self.fillBatch(data_arg2,batches[i],self.X_arg2)
            miniBatch_lbl = self.fillBatch(data_lbl,batches[i],self.label_size)
            e = np.random.normal(0,1,[self.Z,miniBatch_arg1.shape[0]]).astype('float32')

            cost = self.f_grad_shared(x_arg1=miniBatch_arg1.T,x_arg2=miniBatch_arg2.T,\
                    true_class=miniBatch_lbl.T,eps=e)
            self.f_update(self.learning_rate)
            self.lowerbound += self.lowerboundfunction(x_arg1=miniBatch_arg1.T,x_arg2=miniBatch_arg2.T,\
                    true_class=miniBatch_lbl.T,eps=e)

    def getLowerBound(self,data_arg1,data_arg2,data_lbl):
    	"""Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        N_arg1 = len(data_arg1)
        N_arg2 = len(data_arg2)
        N_lbl = len(data_lbl)
        assert N_arg1 == N_arg2 and N_arg1 == N_lbl, 'Hi~man! The arguments should be equal!'
        N = N_arg1

        batches = self.getBatch(N, self.batch_size)
        for i in xrange(0,len(batches)):
            miniBatch_arg1 = self.fillBatch(data_arg1,batches[i],self.X_arg1)
            miniBatch_arg2 = self.fillBatch(data_arg2,batches[i],self.X_arg2)
            miniBatch_lbl = self.fillBatch(data_lbl,batches[i],self.label_size)
            e = np.random.normal(0,1,[self.Z,miniBatch_arg1.shape[0]]).astype('float32')
            lowerbound += self.lowerboundfunction(x_arg1=miniBatch_arg1.T,x_arg2=miniBatch_arg2.T,\
                    true_class=miniBatch_lbl.T,eps=e)

        return lowerbound/N

    def saveParams(self):
        self.params = [p.get_value() for p in self.tparams]
        cPickle.dump(self.params, file('model.params.pkl','w'))
    def loadParams(self):
        import os.path
        if os.path.isfile('./model.params.pkl'):
            print 'loading external parameters'
            self.params = cPickle.load(file('./model.params.pkl','r'))
            self.tparams = [theano.shared(p, name=n) for p,n in zip(self.params, self.params_names)]

options = dict(
    arg1_word_num = 10001,
    arg2_word_num = 10001,
    h_arg1_encoder = 400,
    h_arg2_encoder = 400,
    l_encoder = 400,
    l_decoder = 400,
    h_arg1_decoder = 400,
    h_arg2_decoder = 400,
    z = 20,
    batch_size = 100,
    label_size = 2,
    L = 1,
    clip_c = -1.,
    learning_rate = 0.01,
    max_iter = 1000,
    each_iter = 1,
    is_load = False,
    seed = None,
    mode = 'train',

    train_arg1 = None,
    train_arg2 = None,
    train_lbl = None,
    dev_arg1 = None,
    dev_arg2 = None,
    dev_lbl = None,
    test_arg1 = None,
    test_arg2 = None,
    test_lbl = None
)

def get_f_score(res):
    """return the result given the models, here, 0 is false, and 1 is true"""
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for idx in xrange(len(res)):
        p,t = res[idx]

        if p == t and p == 1:
            true_pos += 1
        elif p == t and p == 0:
            true_neg += 1
        elif p != t and p == 1:
            false_pos += 1
        else:
            false_neg += 1
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-8) * 100.
    pre = true_pos / (true_pos + false_pos + 1e-8) * 100.
    rcl = true_pos / (true_pos + false_neg + 1e-8) * 100.
    f_score = 2. * (pre * rcl) / (pre + rcl + 1e-8)

    return (acc, pre, rcl, f_score)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        options.update(eval(open(sys.argv[1]).read()))

    print options

    # set random seed
    if options['seed'] is not None:
        seed = options['seed']
    else:
        seed = int(time.time())
    print 'seed:\t%s' % seed
    np.random.seed(seed)

    # loading data
    train_arg1 = cPickle.load(file(options['train_arg1'], 'r'))
    train_arg2 = cPickle.load(file(options['train_arg2'], 'r'))
    train_lbl = cPickle.load(file(options['train_lbl'], 'r'))
    dev_arg1 = cPickle.load(file(options['dev_arg1'], 'r'))
    dev_arg2 = cPickle.load(file(options['dev_arg2'], 'r'))
    dev_lbl = cPickle.load(file(options['dev_lbl'], 'r'))
    test_arg1 = cPickle.load(file(options['test_arg1'], 'r'))
    test_arg2 = cPickle.load(file(options['test_arg2'], 'r'))
    test_lbl = cPickle.load(file(options['test_lbl'], 'r'))

    # build model
    encoder = VarNDrr(options['l_encoder'],options['l_decoder'],\
            options['h_arg1_encoder'],options['h_arg1_decoder'],\
            options['h_arg2_encoder'],options['h_arg2_decoder'],\
            options['arg1_word_num'],options['arg2_word_num'],\
            options['z'],options['batch_size'],options['label_size'],\
            L=options['L'],learning_rate=options['learning_rate'],clip_c=options['clip_c'])

    print "Initializing weights and biases"
    encoder.initParams()
    if options['is_load']:
        encoder.loadParams()

    print "Creating Theano functions"
    encoder.createGradientFunctions()

    if options['mode'] == 'train':
        lowerbound = np.array([])
        testlowerbound = np.array([])
        best_dev_model = -1

        N = len(train_arg1)
        begin = time.time()
        # Notice that we did not put any convergence constraints here
        # this is because we want to observe the change of the lower bounds
        # maybe an early-stopping is prefered
        for j in xrange(options['max_iter']):
            encoder.lowerbound = 0
            print 'Iteration:', j
            encoder.iterate(train_arg1, train_arg2, train_lbl)
            end = time.time()
            print("Iteration %d, lower bound = %.2f,"
              " time = %.2fs"
              % (j, encoder.lowerbound/N, end - begin))
            lowerbound = np.append(lowerbound,encoder.lowerbound)
            begin = end

            if np.isnan(encoder.lowerbound):
                print 'Nan dectected'
                break

            if j % options['each_iter'] == 0:
                testlowerbound = np.append(testlowerbound,encoder.getLowerBound(test_arg1, test_arg2, test_lbl))
                print "Calculating test lowerbound, lower bound = %.2f" % testlowerbound[-1]


            dev_res = encoder.getPrediction(dev_arg1, dev_arg2, dev_lbl)
            tun_dev_res = get_f_score(dev_res)
            prefix = "plain"
            if tun_dev_res[-1] > best_dev_model:
                prefix = "best"
                best_dev_model = tun_dev_res[-1]
                encoder.saveParams()
            print prefix + ' dev result: accurancy %s, prediction %s, recall %s, f-score %s' % tun_dev_res

        cPickle.dump(lowerbound, file('train.lower.bound.pkl', 'w'))
        encoder.loadParams()
        encoder.createGradientFunctions()

    tst_res = encoder.getPrediction(test_arg1, test_arg2, test_lbl)
    tun_tst_res = get_f_score(tst_res)
    print 'tst result: accurancy %s, prediction %s, recall %s, f-score %s' % tun_tst_res
