import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def silent_print(string):
    pass

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def bootstrap_auc(y, pred, n_classes, bootstraps = 1000, fold_size = 1000):
    statistics = np.zeros((n_classes, bootstraps))

    for c in range(n_classes):
        df = pd.DataFrame(columns=['y', 'pred'])
        df['y'] = y[:, c]
        df['pred'] = pred[:, c]
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

def class2_95_ci(y, prob):
    # two classes 95% confidence interval
    if y.ndim == 2:
        assert y.shape[1]==2
    else:
        y = np.stack([1-y, y]).T
    if prob.ndim == 2:
        assert prob.shape[1]==2
    else:
        prob = np.stack([1-prob, prob]).T
    stats = bootstrap_auc(y, prob, 2)  # (2, N)
    boot_means = stats[1, :]
    return np.quantile(boot_means, [.025, 0.975])