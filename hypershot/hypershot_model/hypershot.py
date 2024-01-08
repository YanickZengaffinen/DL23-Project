from hypershot_model.models import ResNet12
from hypershot_model.models import ResNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from hypnettorch.mnets.mlp import MLP
from hypnettorch.hnets import HMLP

class Hypershot(nn.Module):
    def __init__(self, kcnn_input_channels, z_length, kcnn_weights,
                       hnet_hidden_layers, hnet_hidden_size, hnet_weights,
                       mnet_hidden_layers, mnet_hidden_size,
                       K, W, i_dim=28, i_cha=1, k_size=4, load_w = False):
        super(Hypershot, self).__init__()
        
        self.kcnn_input_channels = kcnn_input_channels
        self.z_length = z_length
        self.kcnn_weights = kcnn_weights
        self.hnet_hidden_layers = hnet_hidden_layers
        self.hnet_hidden_size = hnet_hidden_size
        self.hnet_weights = hnet_weights
        self.mnet_hidden_layers = mnet_hidden_layers
        self.mnet_hidden_size = mnet_hidden_size
        
        self.i_dim = i_dim
        self.i_cha = i_cha
        self.k_size = k_size
        
        self.K = K
        self.W = W
        self.kernel = None
        self.z_space = None
        
        # self.kcnn = ResNet12(output_size=z_length, hidden_size=64, channels=self.i_cha, dropblock_dropout=0, avg_pool=False)
        # 4 for omniglot
        # 7 for mininet
        self.kcnn = ResNet(z_length=z_length, i_cha=self.i_cha, i_dim=self.i_dim, k_size=self.k_size)
        self.mnet = MLP(n_in=W, n_out=W, hidden_layers=self.mnet_hidden_layers * [self.mnet_hidden_size])
        # K**2 is the size of the kernel
        self.hnet = HMLP(self.mnet.param_shapes, uncond_in_size=W**2, cond_in_size=0, \
                         layers = self.hnet_hidden_layers * [self.hnet_hidden_size],\
                         num_cond_embs=0)
        self.hnet.apply_hyperfan_init(mnet=self.mnet)
        
        if load_w:
            self.hnet.load_state_dict(torch.load(self.hnet_weights))
            self.kcnn.load_state_dict(torch.load(self.kcnn_weights))
            
    
    def compute_kernel(self, X, y):
        """
        Compute Hypershot kernel for a support set X and label y
        It takes the average of the z's for each label as suggested in the Hypershot paper

        Args:
            X (tensor): Support set used to compute the kernel
            y (tensor): corresponding labels

        Returns:
            type: embeddings, kernel
        """
        # Obtain the indices that would sort y_test
        indices = torch.argsort(torch.argmax(y, dim=1))

        # Use the indices to sort the rows of X_test
        sorted_X = X[indices]
        sorted_y = y[indices]

        reshaped_X = sorted_X.view(sorted_X.shape[0], self.i_cha, self.i_dim, self.i_dim)
        nn_X = self.kcnn(reshaped_X)
        
        mean_X = torch.zeros((int(nn_X.shape[0] / self.K), nn_X.shape[1]))
        for i in range(self.W):
            mean_X[i] = torch.mean(nn_X[i*self.K:(i+1)*self.K], dim = 0)
        norm_mean_X = F.normalize(mean_X, p=2, dim=1)
        
        assert(nn_X.shape==(sorted_X.shape[0], self.z_length))
        
        return norm_mean_X, torch.matmul(norm_mean_X, torch.t(norm_mean_X)) 

    def get_s_and_q_sets(self, X, y, trgt_lbls, q_size):
        """
        Computes a support set for data X for classes in y with K sample per classes
        and corresponding query sets of size q_size.

        Args:
            X (tensor): Data used to compute the sets (can contain label you do not want for your sets)
            y (tensor): corresponding labels
            trgt_lbls : the labels that end up in the sets
            q_size: amount of sample per classes in query set

        Returns:
            type: support set, support set labels, query set, query set labels
        """
        s_set = np.zeros((len(trgt_lbls) * self.K, X.shape[1]))
        s_set_lbl = np.zeros((len(trgt_lbls) * self.K))

        q_set = np.zeros((len(trgt_lbls) * q_size, X.shape[1]))
        q_set_lbl = np.zeros((len(trgt_lbls) * q_size))
        for j, l in enumerate(trgt_lbls):
            mask = (y == l)
            masked_data = X[mask]
            masked_lbls = y[mask]
            s_set[j*self.K:(j+1)*self.K] = masked_data[0:self.K]
            s_set_lbl[j*self.K:(j+1)*self.K] = masked_lbls[0:self.K]
            q_set[j*q_size:(j+1)*q_size] = masked_data[self.K:self.K+q_size]
            q_set_lbl[j*q_size:(j+1)*q_size] = masked_lbls[self.K:self.K+q_size]

        s_set = torch.tensor(s_set, requires_grad=True).float()
        s_set_lbl = torch.tensor(s_set_lbl, requires_grad=True).float()
        q_set = torch.tensor(q_set, requires_grad=True).float()
        q_set_lbl = torch.tensor(q_set_lbl, requires_grad=True).float()
        
        return s_set, s_set_lbl, q_set, q_set_lbl

    def get_q_sample_features(self, X):
        """
        Computes the final features used for classification, given a query sample mx

        Args:
            X (tensor): query sample 
            cnn: the cnn trained to compute the desired features
            zs: z space of the support set corresponding to the query sample

        Returns:
            type: final flattened features use by the main network
        """
        X = X.view(-1, self.i_cha, self.i_dim, self.i_dim)
        zs_q = self.kcnn(X)
        zs_q = F.normalize(zs_q, p=2, dim=1)
        zs_q_m = torch.matmul(self.z_space, torch.t(zs_q))
        return torch.t(zs_q_m)

    def compute_sets_and_features(self, X, y, trgt_lbls, q_size, update_kernel):
        s_set, s_set_lbl, q_set, q_set_lbl = self.get_s_and_q_sets(X, y, trgt_lbls, q_size)
        
        z_space, kernel = self.compute_kernel(s_set, s_set_lbl)
        if update_kernel:
            self.z_space = z_space
            self.kernel = kernel
            
        return s_set, s_set_lbl, q_set, q_set_lbl

    def extend_pred_to_nclasses(self, pred, n_c, lbls):
        out = torch.zeros((pred.shape[0], n_c))
        int_lbls = [int(x) for x in lbls]
        for i in range(out.shape[0]):
            out[i][int_lbls] = pred[i]
        return out
    
    def update_kernel(self, s_set, s_set_lbl):
        z_space, kernel = self.compute_kernel(s_set, s_set_lbl)
        self.z_space = z_space
        self.kernel = kernel
    
    def forward(self, x):
        q_features = self.get_q_sample_features(x)
        W = self.hnet(uncond_input=self.kernel.view(1, -1))
        P = self.mnet.forward(q_features, weights=W)
        return P