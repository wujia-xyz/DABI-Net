import time
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import csv

def train(model123, dataloader, optimizer, dev='cpu', class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)

    log('\ttrain')
    model123.train()
    return _train_or_test(model=model123, dataloader=dataloader, optimizer=optimizer, dev=dev,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model123, dataloader, dev='cpu', class_specific=False, log=print, save_logits=False):
    log('\ttest')
    model123.eval()
    return _train_or_test(model=model123, dataloader=dataloader, optimizer=None, dev=dev,
                          class_specific=class_specific, log=log, save_logits=save_logits)

########################################################################

def last_only(model123, log=print):
    for model in model123.module.pnet123:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.add_on_layers.parameters():
            p.requires_grad = False
        model.prototype_vectors.requires_grad = False
    for p in model123.module.last_layer.parameters():
        p.requires_grad = True
    log('\tlast layer')

def warm_only(model123, log=print):
    for model in model123.module.pnet123:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.add_on_layers.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = True
    for p in model123.module.last_layer.parameters():
        p.requires_grad = True  # has gradient but not optimize
    log('\twarm')

def joint(model123, log=print):
    for model in model123.module.pnet123:
        for p in model.features.parameters():
            p.requires_grad = True
        for p in model.add_on_layers.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = True
    for p in model123.module.last_layer.parameters():
        p.requires_grad = True  # has gradient but not optimize
    log('\tjoint')

########################################################################

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

########################################################################

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, save_logits=False, finer_loader=None, dev='cuda'):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_output = []
    total_one_hot_label = []
    confusion_matrix = [0,0,0,0]
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.to(dev)
        target = label.to(dev)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances123, _ = model(input)
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            # only save to csv on test
            if not is_train and save_logits:
                _output_scores = [",".join([str(score) for score in scores.cpu().numpy()]) for scores in output]
                write_file = './training_margin_logits.csv'
                with open(write_file, mode='a') as logit_file:
                    logit_writer = csv.writer(logit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for _index in range(len(_output_scores)):
                        logit_writer.writerow([_output_scores[_index]])
                log(f'Wrote to {write_file}.')

            cluster_cost_l, separation_cost_l, l1_l = [],[],[]
            for mi, min_distances in enumerate(min_distances123):
                num_prototypes = model.module.num_prototypes_l[mi]
                n_protos_cum = sum(model.module.num_prototypes_l[:mi])
                if class_specific:
                    max_dist = (model.module.prototype_shape_l[mi][1]
                                * model.module.prototype_shape_l[mi][2]
                                * model.module.prototype_shape_l[mi][3])
                    prototype_class_identity = model.module.prototype_class_identity[ \
                        n_protos_cum:n_protos_cum+num_prototypes, :]
                    # prototypes_of_correct_class is a tensor of shape [batch_size, num_prototypes]
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(prototype_class_identity[:,label]).to(dev)
                    inverted_distances, _ = torch.max((max_dist - min_distances) * \
                        prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(max_dist - inverted_distances)
                    cluster_cost_l.append(cluster_cost)
                    # print("before change")

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                    separation_cost_l.append(separation_cost)
                    # print("after change")

                    lw = model.module.last_layer.weight[:,n_protos_cum:n_protos_cum+num_prototypes]
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(prototype_class_identity).to(dev)
                        l1 = (lw * l1_mask).norm(p=1)
                    else:
                        l1 = lw.norm(p=1)
                    l1_l.append(l1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            # confusion matrix
            for t_idx, t in enumerate(target):
                if predicted[t_idx] == t and predicted[t_idx] == 1:  # true positive
                    confusion_matrix[0] += 1
                elif t == 0 and predicted[t_idx] == 1:
                    confusion_matrix[1] += 1  # false positives
                elif t == 1 and predicted[t_idx] == 0:
                    confusion_matrix[2] += 1  # false negative
                else:
                    confusion_matrix[3] += 1

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), model.module.num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(output, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)
            # one hot label for AUC

            cluster_cost = sum(cluster_cost_l)
            separation_cost = sum(separation_cost_l)
            l1 = sum(l1_l)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                      + coefs['clst'] * cluster_cost
                      + coefs['sep'] * separation_cost
                      + coefs['l1'] * l1)
            else:
                loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1

            optimizer.zero_grad()
            loss.backward()
            for para in optimizer.param_groups:
                torch.nn.utils.clip_grad.clip_grad_norm_(para['params'], max_norm=2, norm_type=2)
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))

    avg_auc = 0
    for auc_idx in range(model.module.num_classes):
        auc_score = roc_auc_score(np.array(total_one_hot_label)[:, auc_idx],
                                  np.array(total_output)[:, auc_idx])
        avg_auc += auc_score / model.module.num_classes
        log("\tauc score for class {} is: \t\t{}".format(auc_idx, auc_score))

    log('\taccu:        {0}%'.format(n_correct / n_examples * 100))
    e = 1e-10 if confusion_matrix[0]+confusion_matrix[1]==0 else 0
    prec = confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[1]+e) # TP/(TP+FP)
    e = 1e-10 if confusion_matrix[0]+confusion_matrix[2]==0 else 0
    recall = confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[2]+e) # TP/(TP+FN)
    log('\tprecision:   {0}%'.format(prec * 100))
    log('\trecall:      {0}%'.format(recall * 100))
    e = 1e-10 if prec+recall==0 else 0
    log('\tF1 score:    {0}'.format(2 * (prec * recall) / (prec + recall + e)))
    e = 1e-10 if confusion_matrix[1]+confusion_matrix[3]==0 else 0
    specificity = confusion_matrix[3] / (confusion_matrix[1]+confusion_matrix[3]+e) # TN/(TN+FP)
    log('\tspecificity: {0}%'.format(specificity * 100))
    log('\tthe confusion matrix is: {0}'.format(confusion_matrix))

    return avg_auc
