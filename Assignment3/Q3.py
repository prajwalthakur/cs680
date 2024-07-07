# %% [markdown]
# ## Data loading for default of credit card clients dataset 

# %% [markdown]
# To use the data, simply load the pickle file

# %%
import pickle
import numpy as np
import pdb
with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

train_x = train_data[:,:23]
train_y = train_data[:,23] # labels are either 0 or 1
test_x = test_data[:,:23]
test_y = test_data[:,23] # labels are either 0 or 1

# %% [markdown]
# # converting the labels to +-1 

# %%
train_y = 2*train_y - 1
test_y = 2*test_y -1

# %% [markdown]
# # weak classifier

# %% [markdown]
# # Parallel Ada Boost Algorithm-1

# %%
class Adaboost:
    def __init__(self, train_x, train_y,test_x,test_y):
        self.train_x = train_x 
        self.train_y = train_y
        self.test_x =  test_x
        self.test_y = test_y
        self.feature_len = self.train_x.shape[1]
        self.train_len = self.train_x.shape[0]
        self.test_len =  self.test_x.shape[0]
        self.h_weak_train = self.naive_weak_classifier(self.train_x,j=None)
        self.h_weak_test  = self.naive_weak_classifier(self.test_x,j=None)
        self.M_train,self.M_train_normalized = self.cal_M(train_data=True)
        self.M_test,self.M_test_normalized = self.cal_M(train_data=False)
    # h_j = sign(x_j-m_j)
    def naive_weak_classifier(self,X,j=None):
        #X : nxd 
        #j : jth feature <=d if none then comute H_j = n*d matrix
        # return 
        # h_j = nx1 classifier or H_j  = n*d matrix  
        if j != None:
            X_j = X[:,j]
            M_j = np.median(X_j) 
            h_j_temp  = X_j - M_j
            h_j  = np.where(h_j_temp >= 0, 1, -1)[...,np.newaxis]
            return h_j
        else :
            #print(X.shape)
            M = np.median(X,axis=0)[np.newaxis,...]
            #print(M.shape) 
            H_j_temp  = X - M
            H  = np.where(H_j_temp >= 0, 1, -1)
            return H
    def cal_loss(self,w_t):
        H_train = np.multiply(w_t.squeeze(),self.h_weak_train )
        H_test = np.multiply(w_t.squeeze(),self.h_weak_test)
        # calculating the training-error testing-error
        train_pred = np.sum(H_train,axis=1)
        test_pred  = np.sum(H_test,axis=1)
        training_error = (1/self.train_len)*np.sum(np.where(np.multiply(self.train_y,train_pred) <= 0, 1,0))
        testing_error  = (1/self.test_len)*np.sum(np.where(np.multiply(self.test_y,test_pred) <= 0, 1,0))
        # caluclating loss
        train_loss_t = np.sum(np.exp(-self.M_train_normalized@w_t))
        test_loss_t = np.sum(np.exp(-self.M_test_normalized@w_t))
        return training_error,testing_error,train_loss_t,test_loss_t
    
    def cal_M(self,naive_weak_classifier=True,train_data=True):
        if train_data:
            data_x = self.train_x
            target_y = self.train_y 
            H = self.h_weak_train
        else:
            data_x = self.test_x
            target_y = self.test_y
            H = self.h_weak_test

        M = np.multiply(target_y[...,np.newaxis],H).astype(np.float32)
        M_sum_j  = np.sum(M , axis=0)
        M_normalized = M/M_sum_j
        return M,M_normalized
    def adaboost(self,w0 ,p0,max_pass=300, train_data=True):
        if train_data==True:
            data_len = self.train_len
        M = self.M_train_normalized
        w_t = w0
        p_t = p0
        M_positive = np.maximum(M,0)
        M_negative = np.abs(M) - M_positive
        
        train_loss = []
        train_error  =[]
        test_loss = []
        test_error =[]

        for i in range(max_pass):
            training_error_t,testing_error_t,train_loss_t,test_loss_t = self.cal_loss(w_t)
            train_error.append(training_error_t)
            train_loss.append(train_loss_t)
            test_error.append(testing_error_t)
            test_loss.append(test_loss_t)
            print(training_error_t,testing_error_t,train_loss_t,test_loss_t)
            p_t = np.divide(p_t,np.sum(p_t))
            epsi_t = np.transpose(M_negative)@p_t 
            gamma_t = 1- epsi_t
            #print(gamma_t.shape , epsi_t.shape)
            beta_t_temp = np.divide(gamma_t,epsi_t)
            beta_t = np.log(beta_t_temp)
            alpha_t = np.ones((self.feature_len,1)) # Parallel Adaboost
            w_t = w_t + np.multiply(alpha_t,beta_t)
            p_t = np.multiply(p_t,np.exp(-M@(np.multiply(alpha_t,beta_t))))
        return w_t,train_error,train_loss,test_error,test_loss

# %% [markdown]
# # calling Ada BOOST

# %%
adaboost_object = Adaboost(train_x=train_x,train_y=train_y ,test_x=test_x ,test_y=test_y )
w_final,train_error,train_loss,test_error,test_loss = adaboost_object.adaboost(w0 = np.zeros((train_x.shape[1],1)),p0 = np.ones((train_x.shape[0],1)),max_pass=300, train_data=True)
print(w_final.shape)


# %%
w_final

# %%



