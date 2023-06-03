import math
import os
from copy import deepcopy

from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize

from config import config
from datasets import CategoriesSampler,CategoriesSamplerNEW, DataSet
from models.ici import ICI
from utils import get_embedding, mean_confidence_interval, setup_seed

#For displaying results
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import gc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import sys
import itertools

#end

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size(),obj.device)
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


def train_embedding(args):
    setup_seed(2333)
    output_string=[]
    ckpt_root = os.path.join('./ckpt', args.dataset)
    os.makedirs(ckpt_root, exist_ok=True)
    data_root = os.path.join(args.folder, args.dataset)
    from datasets import EmbeddingDataset
    print("source train embeddings")
    source_set = EmbeddingDataset(args.dataset,data_root, args.img_size, 'train')
    source_loader = DataLoader(
        source_set, batch_size=32, shuffle=True)
    test_set = EmbeddingDataset(args.dataset,data_root, args.img_size, 'val')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    elif args.dataset == 'nct':
        num_classes = 4
    elif args.dataset == 'opensrh':
        num_classes = 3
    else:
        num_classes = 64
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    model = model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(120):
        model.train()
        scheduler.step(epoch)
        loss_list = []
        train_acc_list = []
        for images, labels in tqdm(source_loader, ncols=0):
            preds = model(images.to(args.device))
            # print(images.shape,labels.shape)
            # im = transforms.ToPILImage()(images[0])
            # #[[2,1,0],:,:]
                
            # im.save('a.png')
            # break
            loss = criterion(preds, labels.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            train_acc_list.append(preds.max(1)[1].cpu().eq(
                labels).float().mean().item())
        acc = []
        model.eval()
        for images, labels in test_loader:
            # print(images.shape,labels.shape)
            # im = transforms.ToPILImage()(images[0])
            # #[[2,1,0],:,:]
                
            # im.save('b.png')
            # break
            preds = model(images.to(args.device)).detach().cpu()
            preds = torch.argmax(preds, 1).reshape(-1)
            labels = labels.reshape(-1)
            acc += (preds==labels).tolist()
        acc = np.mean(acc)

        
        output_str = 'Epoch:{} Train-loss:{} Train-acc:{} Valid-acc:{}'.format(epoch, str(np.mean(loss_list))[:6], str(
            np.mean(train_acc_list))[:6], str(acc)[:6])
        print(output_str)
        output_string.append(output_str)
        # exit()
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(
                ckpt_root, "res12_epoch{}.pth.tar".format(epoch))
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), os.path.join(ckpt_root,'res12_best.pth.tar'))
    with open(r'OutFiles/nct_train_outfile_{}_{}'.format(args.num_shots,args.unlabel), 'w') as fp:
        for item in output_string:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')


def test(args):
    setup_seed(2333)
    import warnings
    warnings.filterwarnings('ignore')
    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    elif args.dataset == 'nct':
        num_classes = 4
    elif args.dataset == 'opensrh':
        num_classes = 3
    else:
        num_classes = 64
    
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    if args.resume is not None:
        
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
        print("model is loaded with weights")
    model.to(args.device)
    model=model.half()
    model.eval()
    ici = ICI(classifier=args.classifier, num_class=args.num_test_ways,
              step=args.step, reduce=args.embed, d=args.dim)

    data_root = os.path.join(args.folder, args.dataset)
    dataset = DataSet(args.dataset,data_root, 'test', args.img_size)
    sampler = CategoriesSampler(dataset.label, args.num_batches,args.num_test_ways, (args.num_shots, 15, args.unlabel),imb=args.imb)
    
    #sampler = CategoriesSamplerNEW(dataset.label,args.num_batches,args.num_test_ways,args.num_shots,15+args.num_shots+args.unlabel,'dirichlet')
    #sampler = CategoriesSamplerNEW(dataset.label,args.num_batches,args.num_test_ways,args.num_shots,15+args.unlabel,'dirichlet')
    
    
    
    testloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=0, pin_memory=True)
    k = args.num_shots * args.num_test_ways
    loader = tqdm(testloader, ncols=0)
    iterations = math.ceil(args.unlabel/args.step) + \
        2 if args.unlabel != 0 else math.ceil(15/args.step) + 2
    acc_list = [[] for _ in range(iterations)]
    
    plot_ct = 1
    for idxx,(data, indicator) in enumerate(loader):
        targets = torch.arange(args.num_test_ways).repeat(args.num_shots+15+args.unlabel).long()[
            indicator[:args.num_test_ways*(args.num_shots+15+args.unlabel)] != 0]
        data = data[indicator != 0]
#         print(data.shape,data.dtype,data,targets.shape,targets,end="\n")


        data=data.half()
#         print(data.shape,data,targets.shape,targets,end="\n")
#         
#         targets=targets.half()
        train_inputs = data[:k].to(args.device)
        train_targets = targets[:k].cpu().numpy()
        test_inputs = data[k:k+15*args.num_test_ways].to(args.device)
        test_targets = targets[k:k+15*args.num_test_ways].cpu().numpy()
        
#         cpuStats()
#         memReport()
#         
        if args.unlabel != 0:
            unlabel_inputs = data[k+15*args.num_test_ways:]
#         print(train_inputs.shape,test_inputs.shape,unlabel_inputs.shape)

        print(data.shape,targets.shape)
        # from torchvision import transforms
        print(data[0].shape)
        for idx,image in enumerate(data):
            im = transforms.ToPILImage()(image)
            #[[2,1,0],:,:]
            
            im.save('a{}.png'.format(idx))
        exit()
        del data
        del targets
        train_embeddings = get_embedding(model, train_inputs, args.device)
        
        gc.collect()
        del train_inputs
        gc.collect()
        ici.fit(train_embeddings, train_targets)
        
        test_embeddings = get_embedding(model, test_inputs, args.device)
        del test_inputs
        gc.collect()
        if args.unlabel != 0:
#             unlabel_inputs = data[k+15*args.num_test_ways:]
            unlabel_embeddings = get_embedding(
                model, unlabel_inputs, args.device)
        else:
            unlabel_embeddings = None
        acc, predicts = ici.predict(test_embeddings, unlabel_embeddings,
                          True, test_targets)
        
        # Confusion Matrix
        cm = confusion_matrix(test_targets,predicts).astype(int)
        np.savetxt('./StatsImages_'+args.dataset+'/cm_'+str(plot_ct)+'_'+args.savename+'.csv', cm, delimiter=',')
        
        classes = range(args.num_test_ways)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.tick_params(labelsize=12)
        # plt.title(title)
        plt.colorbar(shrink=1.0)
        tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        normalize=False
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label',fontsize=16)
        plt.xlabel('Predicted label',fontsize=16)
        plt.savefig('./StatsImages_'+args.dataset+'/cm_'+str(plot_ct)+'_'+args.savename+'.png', bbox_inches="tight")
        #end
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        
        # Add t-SNE
        n_components = 2
        tsne = TSNE(n_components,perplexity=25.0)
#         print("##############",test_embeddings.shape)
        tsne_result = tsne.fit_transform(test_embeddings)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2':tsne_result[:,1], 'label':test_targets})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1',y='tsne_2',hue='label',data=tsne_result_df,ax=ax,s=120)
        lim = (tsne_result.min()-5,tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_title('Ep'+str(plot_ct))
        ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.0)
        fig.savefig('./StatsImages_'+args.dataset+'/tsne_ep'+str(plot_ct)+'_'+args.savename+'.png')
        # end
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        # Classification Report
        cr = classification_report(test_targets,predicts)
        with open('./StatsImages_'+args.dataset+'/cr_'+str(plot_ct)+'_'+args.savename+'.txt', "w") as text_file:
            text_file.write(cr)
        
        
        
        plot_ct += 1
        
        
        for i in range(min(iterations-1,len(acc))):
            acc_list[i].append(acc[i])
        acc_list[-1].append(acc[-1])
    mean_list = []
    ci_list = []
    for item in acc_list:
        mean, ci = mean_confidence_interval(item)
        mean_list.append(mean)
        ci_list.append(ci)
    print("Test Acc Mean{}".format(
        ' '.join([str(i*100)[:5] for i in mean_list])))
    print("Test Acc ci{}".format(' '.join([str(i*100)[:5] for i in ci_list])))
    with open(r'OutFiles/nct_test_outfile_{}_{}'.format(args.num_shots,args.unlabel), 'w') as fp:
        # write each item on a new line
        fp.write("%s\n" % "Test Acc Mean{}".format(
        ' '.join([str(i*100)[:5] for i in mean_list])))
        fp.write("%s\n" % "Test Acc ci{}".format(' '.join([str(i*100)[:5] for i in ci_list])))
        print('Done')


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    if args.mode == 'train':
        train_embedding(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise NameError


if __name__ == '__main__':
    args = config()
    main(args)
