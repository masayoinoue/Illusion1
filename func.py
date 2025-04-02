#import sys, os
#sys.path.append(os.pardir)  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from common.util import im2col, col2im
from common.gradient import numerical_gradient



# %%
# ----- # ----- # ----- # ----- # ----- #
# for Neural Networks

# make convolution filters
def make_filter(x_wid):
    w_num = 2**x_wid
    w_vrt = np.zeros(w_num*x_wid*x_wid).reshape(w_num, -1, x_wid, x_wid)    # initialization vertical-stripe filter
    w_hrz = np.zeros(w_num*x_wid*x_wid).reshape(w_num, -1, x_wid, x_wid)    # initialization horizontal-stripe filter

    for i1 in range(w_num) :
        for i2 in range(x_wid) :
            j = int(format(i1, '07b')[i2])  #!!! when filter size is (A, A), change "format(i1, '07b')" --> "format(i1, '0Ab')" !!!
            if j == 0:
                j = -1
                
            w_vrt[i1, 0, :, i2 ] = j    # vertical-stripe filter
            w_hrz[i1, 0, i2, : ] = j    # horizontal-stripe filter
            
    w_vrt = w_vrt / (x_wid*x_wid)       # normalization
    w_hrz = w_hrz / (x_wid*x_wid)      

    return w_vrt, w_hrz


# -----
# convolution operations
def convolution(dat, dat_w, x_wid, x_stride, x_pad):
    # change data_shape
    inp = im2col (dat, x_wid, x_wid, x_stride, x_pad) 

    # change filter_shape
    dat_chan = len(dat[0])
    w_num = len(dat_w)
    w = dat_w.reshape(w_num, -1).T
    if dat_chan > 1 :
        w2 = w.copy()
        for i in range(dat_chan - 1 ):
            w = np.concatenate([w, w2], axis=0)

    # compute output
    out = np.dot(inp, w)
    out_wid = int( 1+(len(dat[0,0]) + 2*x_pad - x_wid) / x_stride ) # output size
    out = out.reshape(1, out_wid, out_wid, -1).transpose(0,3,1,2)

    return out


# -----
# relu function
def ReLU(x):
    y = np.maximum(0, x)
    return y



# %%
# ----- # ----- # ----- # ----- # ----- #
# for Training Precess

# for Fully connected layer
class AffineNetwork:
    def __init__(self, x_oi, out_ans, funi = 0.1, n_output = 1):    # Fully_connected_layer initialization
        self.x_oi = x_oi        # output after Convolution and ReLu 
        self.out_ans = out_ans  # answer = target figure
        self.n_input = len(x_oi[0])     # number of filters (channels)
        self.w = np.random.normal(size=(self.n_input), scale=(2.0/n_output)**0.5)   # weight initialization


    def set_weight(self, w):    # set weight_parameter to specific value
        self.w = w


    def eRSS(self, out_pred):   # definition of sum_of_squared_errors for loss_function
        return np.sum((out_pred.ravel() - self.out_ans.ravel()) ** 2) / 2
    

    def predict(self):  # output after fully_connected_layer
        out = self.w[0] * self.x_oi[0, 0]
        for i in range(1, self.n_input):
            out += self.w[i] * self.x_oi[0, i]
        return out 


    def loss(self):     # loss value
        z = self.predict()
        return self.eRSS(z) 


    def grad(self):    # get numerical_gradient
        f = lambda w: self.loss()
        self.grad_w = numerical_gradient(f, self.w)
        return self.grad_w
    

    def paramet(self):  # get weight_parameter
        return self.w
    

    def cyclic(self, w_cycle):  # for model with periodic filters
        w_cyclic = w_cycle + w_cycle
        self.w = self.w * w_cyclic
        for i in range(self.n_input):
            self.x_oi[0, i] = self.x_oi[0, i] * w_cyclic[i] 



# %%
# ----- # ----- # ----- # ----- # ----- #
# for Adam as the optimization algorithm
class myAdam:
    """ (http://arxiv.org/abs/1412.6980v8) """

    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        #print('optimizer Adam start')
        

    def update(self, params, grads, lr=0.001):
        self.lr = lr
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads**2 - self.v)
            
        params -= lr_t * self.m / (np.sqrt(self.v) + 1e-8)
            
        return self.iter



# %%
# ----- # ----- # ----- # ----- # ----- #
# for training process
def Train(epoch, fnet, foptimizer, flr=0.001):
    preloss = fnet.loss()   # loss at initial 
    for i in range(1, epoch+1):  
        grads = fnet.grad()
        k = foptimizer.update(fnet.w, grads, lr=flr)

        if i % 10 == 0 and np.fabs(preloss - fnet.loss()) < preloss*0.01 :  # stop training when loss_value no longer changing
            break
        if i % 10 == 0 :
            preloss = fnet.loss()

    plt.imshow(fnet.predict() , vmin = 0, vmax = 1, cmap = "gray")  # show output image
    plt.colorbar()
    plt.show()
    #np.save(f'{fig_file}-{int(fnet.loss())}-{k}', fnet.paramet() ) # save weight_parameter values
    
