import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.notebook import tqdm

# This class is useful when fine-tuning BERT with IMDB data
class TextSentiment(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, finetune=False):
        super(TextSentiment, self).__init__()
        self.text = data['text'][:]
        self.labels = data['label'][:]
        self.tokenizer = tokenizer
        self.finetune = finetune

    def __getitem__(self, index):
        text = self.text[index]
        if self.finetune:
            inputs = self.tokenizer(text, 
                                  add_special_tokens=True, 
                                  padding='max_length', 
                                  truncation=True, 
                                  return_token_type_ids=False)
        else:
            inputs = self.tokenizer(text, 
                                  add_special_tokens=True, 
                                  padding='longest', 
                                  truncation=True, 
                                  return_token_type_ids=False)

        label = self.labels[index]
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor([label])

        return inputs

    def __len__(self):
        return(len(self.labels))
    

def get_features_layer(model,k,x):
    """
    Inputs: BERT model, index k of layer, input x
    Output: Features of input x at layer k
    """
    default = model.config.output_hidden_states
    model.config.output_hidden_states = True
    if k<0 or k>12:
        print('layer index out of range')
        model.config.output_hidden_states = default
        return
    else:
        with torch.no_grad():
            encoded_layers = model(x)
            sentence_embedding = encoded_layers[1][k][:,0,:]
        model.config.output_hidden_states = default
        return(sentence_embedding)
            

            
def compute_centers_cov(trainloader,model,layer=12,num_classes=2, num_per_class=None, quantity_target=None):
    """
    Computes class means and covariance matrix of the dataset in 'trainloader'
    at the considered layer
    'quantity_target' allows to stop the loop when a specific number of samples is processed.
    This is effectively a subsampling method.
    """
    sample_X = next(iter(trainloader))['input_ids'].cuda()
    center_size = get_features_layer(model,layer,sample_X).size(1)
    
    centers = [torch.zeros(center_size).cuda() for c in range(num_classes)]
    cov_matrix = torch.zeros((center_size, center_size)).cuda()
    
    print(f'Computing class centers; layer {layer}.')
    total=0
    if quantity_target is not None:
        tot_iterations = quantity_target//trainloader.batch_size - 1
    else:
        tot_iterations = len(trainloader) 
    if num_per_class is None:
        cardinals = [0]*num_classes
        for data in tqdm(trainloader, total=tot_iterations):
            batch_X, labels = data['input_ids'].cuda(), data['labels'].cuda()
            features = get_features_layer(model, layer, batch_X)
            for x, c in zip(features, labels):
                centers[c] += x/1000
                cardinals[c] += 1
                total += 1
            if quantity_target is not None and total>=quantity_target:
                break
        for c in range(num_classes):
            centers[c] /= (cardinals[c]/1000)
    else:
        for data in tqdm(trainloader, total=tot_iterations):
            batch_X, labels = data['input_ids'].cuda(), data['labels'].cuda()
            features = get_features_layer(model, layer, batch_X)
            for x, c in zip(features, labels):
                centers[c] += x/num_per_class
                total += 1
            if quantity_target is not None and total>=quantity_target:
                break
                
    print(f'Computing covariance matrix; layer {layer}.')
    total = 0
    for data in tqdm(trainloader, total=tot_iterations):
        batch_X, labels = data['input_ids'].cuda(), data['labels'].cuda()
        features = get_features_layer(model, layer, batch_X)
        for i in range(features.size(0)):
            features[i] = features[i] - centers[labels[0][i].item()]
            total += 1
        cov_matrix += torch.matmul(features.transpose(1,0),features)
        if quantity_target is not None and total>=quantity_target:
            break
    cov_matrix = cov_matrix / total
    return(torch.stack(centers), cov_matrix)


def mahalanobis_distance(batch_X, model, centers, inv_cov_matrix, layer=12):
    """
    Computes Mahalanobis distances of a batch, when the centers and covariance
    matrix are already computed.
    """
    num_classes = len(centers)
    zero_m_feat = [get_features_layer(model,layer,batch_X) - centers[c] 
                   for c in range(num_classes)]
    zero_m_feat = torch.stack(zero_m_feat)
    distances = -torch.matmul(zero_m_feat, inv_cov_matrix).matmul(zero_m_feat.transpose(1,2)).diagonal()
    return(distances.max(1).values)


def mahalanobis_score(batch_X, model, centers, inv_cov_matrix, layer=12, magnitude=0.0012):
    """
    In the absence of batch preprocessing, this function only uses the 'mahalanobis_distance'
    function
    """
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

    