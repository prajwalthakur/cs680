
import pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

train_x = train_data[:,:23]
train_y = train_data[:,23] # labels are either 0 or 1
test_x = test_data[:,:23]
test_y = test_data[:,23] # labels are either 0 or 1
train_y = 2*train_y - 1
test_y = 2*test_y -1



import pandas as pd
df = pd.DataFrame()
class Node:
    def __init__(self,split,loss=None): # split chosen from x values
        self.root_value = split
        self.left = None
        self.right = None
        self.loss = loss
        self.s = None
        self.b = None
        self.missclassification_error = None
    def predict_class(self):
        # In prediction we just count whats in the majority in left and right of the root 
        # weather to choose  if x<split == 1  or  x>split == 1
        #pdb.set_trace()
        pos_labels_count = np.count_nonzero(self.right.root_value.values == 1)
        neg_labels_count = np.count_nonzero(self.right.root_value.values == -1)
        if pos_labels_count>=neg_labels_count:
            self.s = +1
            self.b = -self.root_value
            self.missclassification_error = neg_labels_count
        else:
            self.s = -1
            self.b = +self.root_value
            self.missclassification_error = pos_labels_count
        print("S=",self.s,"b=",self.b,"miss-classification-error=",self.missclassification_error)    
    
class stump(Node):
    def __init__(self):
        self.root  = None
        return 
    
    def fit(self,x, y):
        """
        We train on the (x,y), getting split of single-var x that
        minimizes variance in subregions of y created by x split.
        Return root of decision tree stump
        """
        loss, split = self.find_best_split(x,y)  
        root = Node(split,loss)
        root.left = Node(y[x<split])
        root.right = Node(y[x>=split])
        root.predict_class()
        return root
            
    def find_best_split(self,x,y):
        best_loss = np.inf
        best_split = -1
        #print(f"find_best_split in x={list(x)}")
        for v in x[1:]: # try all possible x values
            left_tree = y[x<v].values
            right_tree = y[x>=v].values
            nl = len(left_tree)
            nr = len(right_tree)
            if nl==0 or nr==0:
                continue
            # variance is same as MSE here
            # weight by proportion on left and right, get avg as loss
            #loss = (np.var(left_tree)*nl + np.var(right_tree)*nr)/2
            ginni_loss_left = 1 - (np.count_nonzero(left_tree == 1)/nl)**2 - (np.count_nonzero(left_tree == -1)/nl)**2
            ginni_loss_right = 1 - (np.count_nonzero(right_tree == 1)/nr)**2 - (np.count_nonzero(right_tree == -1)/nr)**2
            loss = (nl/(nr+nl))*ginni_loss_left +(nr/(nr+nl))*ginni_loss_right
            
            #print(loss)
            #pdb.set_trace()
            #print(f"{left_tree} | {right_tree}    candidate split x ={v:4f} loss {loss:8.1f}")
            if loss < best_loss:
                best_loss = loss
                best_split = v
        return float(best_loss), best_split 
    def fit_all_features(self,X,Y):
        th = [] # list of threshold
        df["y"] = Y.astype(np.int32).tolist()
        for i in range(X.shape[1]):
            print("features=",i,"training")
            df["x"] = X[:,i].astype(np.float32).tolist()
            root_obj= self.fit(df.x, df.y)
            th.append(root_obj) # appending threshold for all features
            #pdb.set_trace()
        return th

stump_classifier = stump()
# df["y"] = train_y.astype(np.int32).tolist()
# df["x"] = train_x[:,11].astype(np.float32).tolist()
stump_model_param_list = stump_classifier.fit_all_features(train_x, train_y)
