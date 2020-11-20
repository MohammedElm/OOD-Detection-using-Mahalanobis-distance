import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm

def get_features_layer(model,k,x):
    """
    Inputs: ResNet model, index k of layer, input x
    Output: Features of input x at layer k
    """
    if k<0 or k>16:
        print('layer index out of range')
        return
    else:
        with torch.no_grad():
            out = F.relu(model.bn1(model.conv1(x)))
            if k == 0:
                return(out)
            else:
                for i in range(1,k):
                    out = model.feature_layer(i-1, out)
                return(out)

            
def compute_centers_cov(trainloader,model,layer=16,num_classes=10, num_per_class=5000):
    """
    Computes class means and covariance matrix of the dataset in 'trainloader'
    at the considered layer
    """
    sample_X = next(iter(trainloader))[0].cuda()
    center_size = get_features_layer(model,layer,sample_X).mean(dim=(2,3)).size(1)
    
    centers = [torch.zeros(center_size).cuda() for c in range(num_classes)]
    cov_matrix = torch.zeros((center_size, center_size)).cuda()
    
    print(f'Computing class centers; layer {layer}.')
    if num_per_class is None:
        cardinals = [0]*num_classes
        for data in tqdm(trainloader):
            batch_X, labels = data[0].cuda(), data[1].cuda()
            features = get_features_layer(model, layer, batch_X).mean(dim=(2,3))
            for x, c in zip(features, labels):
                centers[c] += x
                cardinals[c] += 1
        for c in range(num_classes):
            centers[c] /= cardinals[c]
    else:
        for data in tqdm(trainloader):
            batch_X, labels = data[0].cuda(), data[1].cuda()
            features = get_features_layer(model, layer, batch_X).mean(dim=(2,3))
            for x, c in zip(features, labels):
                centers[c] += x/num_per_class
                
    print(f'Computing covariance matrix; layer {layer}.')
    total_data = len(trainloader)*trainloader.batch_size
    for data in tqdm(trainloader):
        batch_X, labels = data[0].cuda(), data[1].cuda()
        features = get_features_layer(model, layer, batch_X).mean(dim=(2,3))
        for i in range(features.size(0)):
            features[i] = features[i] - centers[labels[i]]
        cov_matrix += torch.matmul(features.transpose(1,0),features)/(total_data)
    return(torch.stack(centers), cov_matrix)


def mahalanobis_distance(batch_X, model, centers, inv_cov_matrix, layer=16):
    """
    Computes Mahalanobis distances of a batch, when the centers and covariance
    matrix are already computed, without preprocessing.
    """
    num_classes = len(centers)
    zero_m_feat = [get_features_layer(model,layer,batch_X).mean(dim=(2,3)) - centers[c] 
                   for c in range(num_classes)]
    zero_m_feat = torch.stack(zero_m_feat)
    distances = -torch.matmul(zero_m_feat, inv_cov_matrix).matmul(zero_m_feat.transpose(1,2)).diagonal()
    return(distances.max(1).values)


def batch_preproc(x, model, centers, inv_cov_matrix, layer=16, magnitude=0.1, id_dataset='cifar10'):
    """
    This function allows to preprocess inputs using the gradient perturbation method used in [1]
    """
    normalize_value = {
    'svhn':(0.5, 0.5, 0.5),
    'cifar10':(0.2023, 0.1994, 0.2010),
    'cifar100':(0.2675, 0.2565, 0.2761)
    }
    model.eval()
    num_classes = len(centers)
    
    #Compute forward output in ResNet to the layer we want
    k=layer
    out = F.relu(model.bn1(model.conv1(x)))
    for i in range(1,k):
        out = model.feature_layer(i-1, out)
    
    #Taking average as suggested in the paper
    f_x= out.mean(dim=(2,3))

    #Finding Closest class
    with torch.no_grad():
        zero_m_feat = [f_x - centers[c] for c in range(num_classes)]
        zero_m_feat = torch.stack(zero_m_feat)
        distances = -torch.matmul(zero_m_feat, inv_cov_matrix).matmul(zero_m_feat.transpose(1,2)).diagonal()
    
    #Closest class
    cl=torch.argmax(distances, dim=1)
    
    #Gradient using autograd
    k=layer
    x.requires_grad = True
    x.grad = None
    
    #Recompute f_x to update the gradients
    out = F.relu(model.bn1(model.conv1(x)))
    for i in range(1,k):
        out = model.feature_layer(i-1, out)
    f_x= out.mean(dim=(2,3))
    f_x_c_0 = f_x - centers[cl]
    #Mahalanobis distance w.r.t the closest class
    distance_cl= torch.matmul(f_x_c_0, inv_cov_matrix).matmul(f_x_c_0.transpose(0,1)).diagonal()
    distance_cl.backward(torch.ones((x.size(0))).cuda())
    
    # Noise
    # Normalizing the gradient to binary in {0, 1}
    gradient = x.grad
    gradient = torch.ge(gradient, 0)
    gradient = (gradient.float() - 0.5) * 2 # to {-1,1}
    # Normalizing the gradient to the same space of image
    norm_val = normalize_value[id_dataset]
    gradient[:][0][0] = gradient[:][0][0]/(norm_val[0])
    gradient[:][0][1] = gradient[:][0][1]/(norm_val[1])
    gradient[:][0][2] = gradient[:][0][2]/(norm_val[2])
    # Adding small perturbations to images
    x = torch.add(x,  -magnitude, gradient)
    
    x.detach()
    
    return x

def mahalanobis_score(batch_X, model, centers, inv_cov_matrix, layer=16, magnitude=0.1, id_dataset='cifar10'):
    """
    Function used to put together the two previous functions.
    Computes Mahalanobis score with input preprocessing
    """
    batch_X = batch_preproc(batch_X, model, centers, inv_cov_matrix, layer, magnitude, id_dataset)
    return(mahalanobis_distance(batch_X, model, centers, inv_cov_matrix, layer))


#Evaluation metrics
def metrics_eval(scores_ID,scores_OOD):
    
        ''' Inputs are Lists '''

        scores_ID=np.array(scores_ID)
        scores_OOD=np.array(scores_OOD)
        
        #tnr at 95 tpr
        scores_ID.sort()
        scores_OOD.sort()
        end = np.max([np.max(scores_ID), np.max(scores_OOD)])
        start = np.min([np.min(scores_ID),np.min(scores_OOD)])
        
        num_k = scores_ID.shape[0]
        num_n = scores_OOD.shape[0]
        
        tp = -np.ones([num_k+num_n+1], dtype=int)
        fp = -np.ones([num_k+num_n+1], dtype=int)
        tp[0], fp[0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[l+1:] = tp[l]
                fp[l+1:] = np.arange(fp[l]-1, -1, -1)
                break
            elif n == num_n:
                tp[l+1:] = np.arange(tp[l]-1, -1, -1)
                fp[l+1:] = fp[l]
                break
            else:
                if scores_OOD[n] < scores_ID[k]:
                    n += 1
                    tp[l+1] = tp[l]
                    fp[l+1] = fp[l] - 1
                else:
                    k += 1
                    tp[l+1] = tp[l] - 1
                    fp[l+1] = fp[l]
        tpr95_pos = np.abs(tp / num_k - .95).argmin()
        tnr_at_tpr95 =100. *( 1. - fp[tpr95_pos] / num_n)
        
        #auroc
        tpr = np.concatenate([[1.], tp/tp[0], [0.]])
        fpr = np.concatenate([[1.], fp/fp[0], [0.]])
        results = -np.trapz(1.-fpr, tpr)
        auroc=100.*results
        
        # DTACC
        results = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()
        dtacc=100.*results
        
        # AUIN
        denom = tp+fp
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp/denom, [0.]])
        results = -np.trapz(pin[pin_ind], tpr[pin_ind])
        auin=100.*results
        
        #auout
        denom = tp[0]-tp+fp[0]-fp
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
        results= np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        auout= 100.*results
        
        return {'auroc': auroc, 'auprIN':auin, 'auprOUT':auout, 'detection accuracy':dtacc, 'tnr at 95 tpr': tnr_at_tpr95}
    