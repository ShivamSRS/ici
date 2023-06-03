import os
import os.path as osp
import pickle
import csv
import collections

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from opensrhimproc import get_srh_base_aug,process_read_im,get_srh_vit_base_aug


class DataSet(Dataset):

    def __init__(self,name, data_root, setname, img_size):
        self.img_size = img_size
        csv_path = osp.join(data_root, setname + '.csv')
        self.name = name
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
#         print(lines)
#         exit()
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        if setname=='test' or setname=='val':
            if self.name=="opensrh":
                self.transform = transforms.Compose(get_srh_base_aug() + [
                                                   transforms.Resize((img_size, img_size)),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
            else:
                self.transform = transforms.Compose([
                                                   transforms.Resize((img_size, img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
        else:
            if self.name=="opensrh":
                self.transform = transforms.Compose(get_srh_base_aug() + [
                                                transforms.RandomResizedCrop((img_size, img_size)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            ])
            else:
                self.transform = transforms.Compose([
                                                transforms.RandomResizedCrop((img_size, img_size)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            ])
                
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        path, label = self.data[i], self.label[i]
        
#         print(self.transform)
        if self.name=="opensrh":
            im_temp =process_read_im(path)
            image = self.transform(im_temp)
        else:
            im_temp =Image.open(path)
            image = self.transform(im_temp.convert('RGB'))
#         print(image.shape)
#         exit()
        return image, 1


#class CategoriesSamplerOLD():
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per,imb=False):
        self.n_batch = n_batch #num_batches
        self.n_cls = n_cls # test_ways
        self.n_per = np.sum(n_per) # num_per_class
        self.number_distract = n_per[-1]
        
        self.non_unl = n_per[0]+n_per[1]
        
        self.unl = n_per[2]
        
        if self.unl == 15+7+10:
            self.distribution = [15+7+10,15+7,15,15-10,15-14]
            #opensrh
            #mening 52626
            #metast    34091
            #self.distribution = [15+5+5,15+5,15,15-5,15-10]
        elif self.unl == 50+25+40:
            #self.distribution = [50+25+40,50+24,50,50-40,50-49]
            
            self.distribution = [106,73,50,33,3]
            
            #self.distribution = [50+22+35,50+23,50,50-35,50-45]
        elif self.unl == 100+40+99:
            #self.distribution = [100+40+99,100+40,100,100-80,100-99]
            
            self.distribution = [213,147,100,33,7]
            
            #self.distribution = [100+48+75,100+47,100,100-75,100-95]
        elif self.unl == 15:
            
            self.distribution = [5,6,4]
        elif self.unl == 30:
            
            self.distribution = [10,12,8]
        elif self.unl == 50:
            #0.33199, 0.399 0.2686
            self.distribution = [16,20,14]
        elif self.unl == 100:
            #0.33199, 0.399 0.2686
            self.distribution = [33,40,27]
            
        
        self.imb = imb

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            indicator_batch = []
            classes = torch.randperm(len(self.m_ind))
            trad_classes = classes[:self.n_cls]
            class_ct = 0
            for c in trad_classes:
                #print("c =", c, " and class_ct = ", class_ct)
                l = self.m_ind[c]
                #print("l = ", l)
                #print("L")
                #print(l)
                #print("self.n_per")
                #print(self.n_per)
                
                if self.imb == 1:
                    
                    n_per = self.distribution[class_ct] + self.non_unl
                    # print(self.distribution,self.distribution[class_ct],n_per)
                else:
                    n_per = self.n_per
                class_ct += 1
                print(self.distribution,self.distribution[class_ct],n_per)
#                 exit()
                pos = torch.randperm(len(l))[:n_per]
                #print("POS LEN = ", pos.shape)
                cls_batch = l[pos]
                #print("cls batch =", cls_batch)
                
                cls_indicator = np.zeros(n_per)
                cls_indicator[:cls_batch.shape[0]] = 1
                
                
                if cls_batch.shape[0] != self.n_per:
                    cls_batch = torch.cat([cls_batch, -1*torch.ones([self.n_per-cls_batch.shape[0]]).long()], 0)           
                batch.append(cls_batch)
#                 print("CLS BATCH")
#                 print(cls_batch)
#                 print(len(cls_batch))
#                 print("CLS INDICATOR")
#                 print(cls_indicator)
#                 print(len(cls_indicator))
                indicator_batch.append(cls_indicator)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


filenameToPILImage = lambda x: Image.open(x).convert('RGB')

def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels


class EmbeddingDataset(Dataset):

    def __init__(self, dataroot, img_size, type = 'train'):
        self.img_size = img_size
        # Transformations to the image
        if type=='train':
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.RandomCrop(img_size, padding=8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        
        self.ImagesDir = os.path.join(dataroot,'images')
        self.data = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))

        self.data = collections.OrderedDict(sorted(self.data.items()))
        keys = list(self.data.keys())
        self.classes_dict = {keys[i]:i  for i in range(len(keys))} # map NLabel to id(0-99)

        self.Files = []
        self.belong = []

        for c in range(len(keys)):
            num = 0
            num_train = int(len(self.data[keys[c]]) * 9 / 10)
            for file in self.data[keys[c]]:
                if type == 'train' and num <= num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                elif type=='val' and num>num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                num = num+1


        self.__size = len(self.Files)

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(self.ImagesDir,str(File))
        try:
            images = self.transform(path)
        except RuntimeError:
            import pdb;pdb.set_trace()
        return images,c

    def __len__(self):
        return self.__size


####
#### Imbalanced Part
####

import torch
import numpy as np
import math


class CategoriesSamplerNEW():
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                n_cls : Number of classification ways (n_ways)
                s_shot : Support shot
                q_shot : Query shot (balanced)
                balanced : 'balanced': Balanced query class distribution: Standard class balanced Few-Shot setting
                           'dirichlet': Dirichlet's distribution over query data: Realisatic class imbalanced Few-Shot setting
                alpha : Dirichlet's concentration parameter
            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]
                        Where :
                            Support data and labels class sequence is :
                                [a b c d e a b c d e a b c d e ...]
                             Query data and labels class sequence is :
                               [a a a a a a a a b b b b b b b c c c c c d d d d d e e e e e ...]
    """
    def __init__(self, label, n_batch, n_cls, s_shot, q_shot, balanced, alpha = 2):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.s_shot = s_shot
        self.q_shot = q_shot
        self.balanced = balanced
        self.alpha = alpha

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            support = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
                support.append(l[pos])
            support = torch.stack(support).t().reshape(-1)
            # support = torch.stack(support).reshape(-1)

            query = []
            alpha = self.alpha * np.ones(self.n_cls)
            if self.balanced == 'balanced':
                query_samples = np.repeat(self.q_shot, self.n_cls)
            else:
                query_samples = get_dirichlet_query_dist(alpha, 1, self.n_cls, self.n_cls * self.q_shot)[0]

            for c, nb_shot in zip(classes, query_samples):
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:nb_shot]  # sample n_per data index of this class
                query.append(l[pos])
            query = torch.cat(query)

            batch = torch.cat([support, query])

            yield batch


def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)

# import os
# import os.path as osp
# import pickle
# import csv
# import collections

# import numpy as np
# import PIL.Image as Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torchvision.transforms import Compose, Normalize, Resize, ToTensor
# from opensrhimproc import get_srh_base_aug,process_read_im,get_srh_vit_base_aug


# class DataSet(Dataset):

#     def __init__(self,name, data_root, setname, img_size):
#         self.img_size = img_size
#         csv_path = osp.join(data_root, setname + '.csv')
#         self.name = name
#         lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
# #         print(lines)
# #         exit()
#         data = []
#         label = []
#         lb = -1

#         self.wnids = []

#         for l in lines:
#             name, wnid = l.split(',')
#             path = osp.join(data_root, 'images', name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
#             data.append(path)
#             label.append(lb)

#         self.data = data
#         self.label = label
        
#         if setname=='test' or setname=='val':
#             if self.name=="opensrh":
#                 self.transform = transforms.Compose(get_srh_base_aug() + [
#                                                    transforms.Resize((img_size, img_size)),
#                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#             else:
#                 self.transform = transforms.Compose([
#                                                    transforms.Resize((img_size, img_size)),
#                                                    transforms.ToTensor(),
#                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#         else:
#             if self.name=="opensrh":
#                 self.transform = transforms.Compose(get_srh_base_aug() + [
#                                                 transforms.RandomResizedCrop((img_size, img_size)),
#                                                 transforms.RandomHorizontalFlip(),
#                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                             ])
#             else:
#                 self.transform = transforms.Compose([
#                                                 transforms.RandomResizedCrop((img_size, img_size)),
#                                                 transforms.RandomHorizontalFlip(),
#                                                 transforms.ToTensor(),
#                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                             ])
                
            
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         if i == -1:
#             return torch.zeros([3, self.img_size, self.img_size]), 0
#         path, label = self.data[i], self.label[i]
        
#         # print(self.transform)
#         if self.name=="opensrh":
#             im_temp =process_read_im(path)
#             image = self.transform(im_temp)
#         else:
#             im_temp =Image.open(path)
#             image = self.transform(im_temp.convert('RGB'))
# #         print(image.shape)
# #         exit()
#         return image, 1


# #class CategoriesSamplerOLD():
# class CategoriesSampler():

#     def __init__(self, label, n_batch, n_cls, n_per,imb=False):
#         self.n_batch = n_batch #num_batches
#         self.n_cls = n_cls # test_ways
#         self.n_per = np.sum(n_per) # num_per_class
#         self.number_distract = n_per[-1]
        
#         self.unl = n_per[2]
        
#         self.imb = imb

#         label = np.array(label)
#         self.m_ind = []
#         for i in range(max(label) + 1):
#             ind = np.argwhere(label == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)

#     def __len__(self):
#         return self.n_batch
    
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             batch = []
#             indicator_batch = []
#             classes = torch.randperm(len(self.m_ind))
#             trad_classes = classes[:self.n_cls]
#             class_ct = 0
#             for c in trad_classes:
#                 l = self.m_ind[c]
#                 #print("L")
#                 #print(l)
#                 #print("self.n_per")
#                 #print(self.n_per)
                
#                 if self.imb == 1:
#                     n_per = self.n_per - class_ct*int(self.unl/7)
#                 else:
#                     n_per = self.n_per
#                 class_ct += 1
                
#                 pos = torch.randperm(len(l))[:n_per]
#                 cls_batch = l[pos]
#                 cls_indicator = np.zeros(n_per)
#                 cls_indicator[:cls_batch.shape[0]] = 1
#                 if cls_batch.shape[0] != self.n_per:
#                     cls_batch = torch.cat([cls_batch, -1*torch.ones([self.n_per-cls_batch.shape[0]]).long()], 0)
#                 batch.append(cls_batch)
#                 indicator_batch.append(cls_indicator)
#             batch = torch.stack(batch).t().reshape(-1)
#             yield batch


# filenameToPILImage = lambda x: Image.open(x).convert('RGB')

# def loadSplit(splitFile):
#             dictLabels = {}
#             with open(splitFile) as csvfile:
#                 csvreader = csv.reader(csvfile, delimiter=',')
#                 next(csvreader, None)
#                 for i,row in enumerate(csvreader):
#                     filename = row[0]
#                     label = row[1]
#                     if label in dictLabels.keys():
#                         dictLabels[label].append(filename)
#                     else:
#                         dictLabels[label] = [filename]
#             return dictLabels


# class EmbeddingDataset(Dataset):

#     def __init__(self,name, dataroot, img_size, type = 'train'):
#         self.img_size = img_size
#         self.name = name
#         # print(self.img_size)
#         # Transformations to the image
#         if type=='train':
#             if self.name=="opensrh":
#                 self.transform = transforms.Compose(get_srh_base_aug() + [
#                                                 transforms.Resize((img_size, img_size)),
#                                                     transforms.RandomCrop(img_size, padding=8),
#                                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                                                     transforms.RandomHorizontalFlip(),
#                                                     # transforms.ToTensor(),
#                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                             ])
#             else:
#                 self.transform = transforms.Compose([filenameToPILImage,
#                                                     transforms.Resize((img_size, img_size)),
#                                                     transforms.RandomCrop(img_size, padding=8),
#                                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                                                     transforms.RandomHorizontalFlip(),
#                                                     transforms.ToTensor(),
#                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                                     ])
#         else:
#             if self.name=="opensrh":
#                 self.transform = transforms.Compose(get_srh_base_aug() + [
#                                                    transforms.Resize((img_size, img_size)),
#                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#             else:
#                 self.transform = transforms.Compose([filenameToPILImage,
#                                                 transforms.Resize((img_size, img_size)),
#                                                 transforms.ToTensor(),
#                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                                 ])

        
#         self.ImagesDir = os.path.join(dataroot,'images')
#         self.data = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))

#         self.data = collections.OrderedDict(sorted(self.data.items()))
#         keys = list(self.data.keys())
#         self.classes_dict = {keys[i]:i  for i in range(len(keys))} # map NLabel to id(0-99)

#         self.Files = []
#         self.belong = []

#         for c in range(len(keys)):
#             num = 0
#             num_train = int(len(self.data[keys[c]]) * 9 / 10)
#             for file in self.data[keys[c]]:
#                 if type == 'train' and num <= num_train:
#                     self.Files.append(file)
#                     self.belong.append(c)
#                 elif type=='val' and num>num_train:
#                     self.Files.append(file)
#                     self.belong.append(c)
#                 num = num+1


#         self.__size = len(self.Files)

#     def __getitem__(self, index):

#         c = self.belong[index]
#         File = self.Files[index]

        
#         # print(self.transform)

#         path = os.path.join(self.ImagesDir,str(File))
#         try:
#             if self.name=="opensrh":
#                 im_temp =process_read_im(path)
#                 images = self.transform(im_temp)
#                 # im = transforms.ToPILImage()(images[0])
#                 #[[2,1,0],:,:]
                
#                 # im.save('a.png')
#             else:
#                 im_temp =Image.open(path)
#                 images = self.transform(path)
                
        
        
#         except RuntimeError:
#             import pdb;pdb.set_trace()
#         # exit()
#         return images,c

#     def __len__(self):
#         return self.__size


# ####
# #### Imbalanced Part
# ####

# import torch
# import numpy as np
# import math


# class CategoriesSamplerNEW():
#     """
#             CategorySampler
#             inputs:
#                 label : All labels of dataset
#                 n_batch : Number of batches to load
#                 n_cls : Number of classification ways (n_ways)
#                 s_shot : Support shot
#                 q_shot : Query shot (balanced)
#                 balanced : 'balanced': Balanced query class distribution: Standard class balanced Few-Shot setting
#                            'dirichlet': Dirichlet's distribution over query data: Realisatic class imbalanced Few-Shot setting
#                 alpha : Dirichlet's concentration parameter
#             returns :
#                 sampler : CategoriesSampler object that will yield batch when iterated
#                 When iterated returns : batch
#                         data : torch.tensor [n_support + n_query, channel, H, W]
#                                [support_data, query_data]
#                         labels : torch.tensor [n_support + n_query]
#                                [support_labels, query_labels]
#                         Where :
#                             Support data and labels class sequence is :
#                                 [a b c d e a b c d e a b c d e ...]
#                              Query data and labels class sequence is :
#                                [a a a a a a a a b b b b b b b c c c c c d d d d d e e e e e ...]
#     """
#     def __init__(self, label, n_batch, n_cls, s_shot, q_shot, balanced, alpha = 2):
#         self.n_batch = n_batch  # the number of iterations in the dataloader
#         self.n_cls = n_cls
#         self.s_shot = s_shot
#         self.q_shot = q_shot
#         self.balanced = balanced
#         self.alpha = alpha

#         label = np.array(label)  # all data label
#         self.m_ind = []  # the data index of each class
#         for i in range(max(label) + 1):
#             ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)

#     def __len__(self):
#         return self.n_batch

#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             support = []
#             classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
#             for c in classes:
#                 l = self.m_ind[c]  # all data indexs of this class
#                 pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
#                 support.append(l[pos])
#             support = torch.stack(support).t().reshape(-1)
#             # support = torch.stack(support).reshape(-1)

#             query = []
#             alpha = self.alpha * np.ones(self.n_cls)
#             if self.balanced == 'balanced':
#                 query_samples = np.repeat(self.q_shot, self.n_cls)
#             else:
#                 query_samples = get_dirichlet_query_dist(alpha, 1, self.n_cls, self.n_cls * self.q_shot)[0]

#             for c, nb_shot in zip(classes, query_samples):
#                 l = self.m_ind[c]  # all data indexs of this class
#                 pos = torch.randperm(len(l))[:nb_shot]  # sample n_per data index of this class
#                 query.append(l[pos])
#             query = torch.cat(query)

#             batch = torch.cat([support, query])

#             yield batch


# def convert_prob_to_samples(prob, q_shot):
#     prob = prob * q_shot
#     for i in range(len(prob)):
#         if sum(np.round(prob[i])) > q_shot:
#             while sum(np.round(prob[i])) != q_shot:
#                 idx = 0
#                 for j in range(len(prob[i])):
#                     frac, whole = math.modf(prob[i, j])
#                     if j == 0:
#                         frac_clos = abs(frac - 0.5)
#                     else:
#                         if abs(frac - 0.5) < frac_clos:
#                             idx = j
#                             frac_clos = abs(frac - 0.5)
#                 prob[i, idx] = np.floor(prob[i, idx])
#             prob[i] = np.round(prob[i])
#         elif sum(np.round(prob[i])) < q_shot:
#             while sum(np.round(prob[i])) != q_shot:
#                 idx = 0
#                 for j in range(len(prob[i])):
#                     frac, whole = math.modf(prob[i, j])
#                     if j == 0:
#                         frac_clos = abs(frac - 0.5)
#                     else:
#                         if abs(frac - 0.5) < frac_clos:
#                             idx = j
#                             frac_clos = abs(frac - 0.5)
#                 prob[i, idx] = np.ceil(prob[i, idx])
#             prob[i] = np.round(prob[i])
#         else:
#             prob[i] = np.round(prob[i])
#     return prob.astype(int)


# def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
#     alpha = np.full(n_ways, alpha)
#     prob_dist = np.random.dirichlet(alpha, n_tasks)
#     return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)