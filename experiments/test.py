import numpy as np
import sklearn.metrics as metrics
import torch


def test(net, testloader, loadname=None):
    # checkpoints = torch.load(loadname)
    # net.load_state_dict(checkpoints)
    hs = np.zeros((0, 64))
    corret, total = 0, 0
    predicted_tot, labels_tot, prob_tot = np.array([]), np.array([]), []
    net.eval()
    for datas, labels in testloader:
        datas = datas.float().cuda()
        labels = labels.cuda()
        h, outputs = net(datas, seq_out=True)
        h = torch.nn.functional.normalize(h.mean(2), 1)
        prob = torch.nn.functional.softmax(outputs, 1).detach().cpu().numpy()
        _, predicted = torch.max(outputs.data, 1)

        labels = labels.long()
        total += labels.size(0)
        corret += (predicted == labels).sum()

        predicted_tot = np.append(predicted_tot, predicted.cpu().numpy())  # .cpu().numpy())
        labels_tot = np.append(labels_tot, labels.cpu().numpy())  # labels.cpu().squeeze(1).numpy())
        prob_tot.append(prob)
        hs = np.append(hs, h.cpu().detach().numpy(), axis=0)
    hs_tot = np.array(hs)
    labels_tot = labels_tot

    if loadname:
        np.save(loadname + '.npy', {'h': hs_tot, 'l': labels_tot})

    prob_tot = np.concatenate(prob_tot)
    labels_tot_oneh = torch.nn.functional.one_hot(torch.tensor(labels_tot).long()).float().numpy()
    # evaluation
    f1_macro = metrics.f1_score(labels_tot, predicted_tot, average='macro')
    f1_macro_weight = metrics.f1_score(labels_tot, predicted_tot, average='weighted')
    acc = metrics.accuracy_score(labels_tot, predicted_tot)
    precision = metrics.precision_score(labels_tot, predicted_tot, average='macro')
    recall = metrics.recall_score(labels_tot, predicted_tot, average='macro')
    # print(labels_tot_oneh.shape)
    # print(prob_tot.shape)
    # print(labels_tot_oneh[0])
    # print(prob_tot[0])
    # print(labels_tot_oneh[1])
    # print(prob_tot[1])
    # print(labels_tot_oneh[2])
    # print(prob_tot[2])
    # auc = metrics.roc_auc_score(labels_tot_oneh, prob_tot)

    # print("\t Weighted - average Test fscore:\t{:.3f} ".format(metrics.f1_score(labels_tot,
    #                                                                             predicted_tot,
    #                                                                             average='weighted')))
    # print("\t Macro - average Test fscore:\t{:.3f} ".format(metrics.f1_score(labels_tot,
    #                                                                          predicted_tot,
    #                                                                          average='macro')))
    # print("\t Micro - average Test fscore:\t{:.3f} ".format(metrics.f1_score(labels_tot,
    #                                                                          predicted_tot,
    #                                                                          average='micro')))
    # print("\t Macro - average Test precision:\t{:.3f} ".format(metrics.precision_score(labels_tot,
    #                                                                                    predicted_tot,
    #                                                                                    average='macro')))
    # print("\t Macro - average Test recall:\t{:.3f} ".format(metrics.recall_score(labels_tot,
    #                                                                              predicted_tot,
    #                                                                              average='macro')))
    # print("\t Average Test accuracy:\t{:.3f} ".format(metrics.accuracy_score(labels_tot,
    #                                                                          predicted_tot)))
    # print("\t Average Test auc:\t{:.3f} ".format(metrics.roc_auc_score(labels_tot_oneh,
    #                                                                    prob_tot)))
    # for i in range(0, 3):
    #     print("\tTest fscore of %d th motion: %f " % (int(i), metrics.f1_score(labels_tot == i, predicted_tot == i)))
    return f1_macro, acc, precision, recall, f1_macro_weight#, auc
