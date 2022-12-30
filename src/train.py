import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
from tokenizer import Tokenizer
import random
import time
import sys
from math import ceil

from utils.config import Config
from utils.utils_func import *
from utils.utils_data import DLoader
from models.model import BERT



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # define tokenizer
        self.tokenizer = Tokenizer(self.config)

        # dataloader and split dataset
        torch.manual_seed(999)  # for reproducibility
        self.dataset = DLoader(load_dataset(self.data_path), self.tokenizer, self.config)
        data_size = len(self.dataset)
        train_size = int(data_size * 0.8)
        val_size = int(data_size * 0.1)
        test_size = data_size - train_size - val_size

        self.trainset, self.valset, self.testset = random_split(self.dataset, [train_size, val_size, test_size])
        if self.mode == 'train':
            self.dataset = {'train': self.trainset, 'val': self.valset, 'test': self.testset}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {'test': self.testset}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = BERT(self.config, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
        if self.mode == 'train':
            total_steps = ceil(len(self.dataloaders['train'].dataset) / self.batch_size) * self.epochs
            pct_start = 100 / total_steps
            final_div_factor = self.lr / 25 / 2.5e-6
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_val_acc = 0 if not self.continuous else self.loss_data['best_val_acc']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']
        self.loss_data = {
            'loss': {'train': [], 'val': []}, \
            'acc': {'train': [], 'val': []}
            }

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss, total_acc = 0, 0
                for i, (x, label, attn_mask) in enumerate(self.dataloaders[phase]):
                    batch_size = x.size(0)
                    x, label, attn_mask = x.to(self.device), label.to(self.device), attn_mask.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output = self.model(x, attn_mask)
                        loss = self.criterion(output, label)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                        acc = torch.sum(torch.argmax(output, dim=-1).detach().cpu() == label.detach().cpu()) / batch_size

                    total_loss += loss.item() * batch_size
                    total_acc += acc * batch_size

                    if i % 30 == 0:
                        print('Epoch {}: {}/{} step loss: {}, acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), acc))
                
                dataset_len = len(self.dataloaders[phase].dataset)
                epoch_loss = total_loss / dataset_len
                epoch_acc = total_acc / dataset_len
                print('{} loss: {:4f}, acc: {}\n'.format(phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    self.loss_data['loss']['train'].append(epoch_loss)
                    self.loss_data['acc']['train'].append(epoch_acc)

                elif phase == 'val':
                    self.loss_data['loss']['val'].append(epoch_loss)
                    self.loss_data['acc']['val'].append(epoch_acc)
            
                    # save best model
                    early_stop += 1
                    if  epoch_acc > best_val_acc:
                        early_stop = 0
                        best_val_acc = epoch_acc
                        best_epoch = best_epoch_info + epoch + 1
                        self.loss_data['best_epoch'] = best_epoch
                        self.loss_data['best_val_acc'] = best_val_acc
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val acc: {:4f}, best epoch: {:d}\n'.format(best_val_acc, best_epoch))

        return self.loss_data

    
    def test(self, result_num):
        phase = 'test'
        all_txt, all_gt, all_pred, ids = [], [], [], set()

        if result_num > len(self.dataloaders[phase].dataset):
            print('The number of results that you want to see are larger than total test set')
            sys.exit()

        from tqdm import tqdm

        with torch.no_grad():
            total_loss, total_acc = 0, 0
            self.model.eval()

            for x, label, attn_mask in tqdm(self.dataloaders[phase]):
                # predict and measure statistics
                batch_size = x.size(0)
                x, label, attn_mask = x.to(self.device), label.to(self.device), attn_mask.to(self.device)

                output = self.model(x, attn_mask)
                loss = self.criterion(output, label)

                output = torch.argmax(output, dim=-1).detach().cpu()
                label = label.detach().cpu()
                acc = torch.sum(output == label) / batch_size

                total_loss += loss.item() * batch_size
                total_acc += acc * batch_size

                # collect all results
                all_txt.append(x.detach().cpu())
                all_gt.append(label)
                all_pred.append(output)

        # print final statistics
        dataset_len = len(self.dataloaders[phase].dataset)
        epoch_loss = total_loss / dataset_len
        epoch_acc = total_acc / dataset_len
        print('{} loss: {:4f}, acc: {}\n'.format(phase, epoch_loss, epoch_acc))

        # print predicted samples
        all_txt = torch.cat(all_txt, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        all_pred = torch.cat(all_pred, dim=0)

        while len(ids) != result_num:
            ids.add(random.randrange(all_txt.size(0)))
        ids = list(ids)

        for id in ids:
            l, p = label_mapping(all_gt[id]), label_mapping(all_pred[id])
            print('*'*100)
            print("review: " + self.tokenizer.decode(all_txt[id].tolist()))
            print("gt    : " + l)
            print("pred  : " + p)
            print('*'*100 + '\n\n')
            
        # other statistics
        class_name = ['negative', 'mediocre', 'positive']
        print(classification_report(all_gt, all_pred, target_names=class_name))
        
        # visualization the entire statistics
        cm = confusion_matrix(all_gt, all_pred)
        cm = pd.DataFrame(cm, index=class_name, columns=class_name)
        plt.figure(figsize=(12, 8))
        plt.imshow(cm, cmap='Blues', interpolation=None)
        plt.title('Visualized Statistics', fontsize=20)
        plt.xlabel('Prediction', fontsize=20)
        plt.ylabel('True', fontsize=20)
        plt.xticks(range(len(class_name)), class_name, fontsize=17, rotation=30)
        plt.yticks(range(len(class_name)), class_name, fontsize=17, rotation=30)
        plt.colorbar()
        
        for i in range(cm.shape[1]):
            for j in range(cm.shape[0]):
                plt.text(i, j, round(cm.iloc[j, i], 1), ha='center', va='center', fontsize=17)
                
        plt.savefig(self.base_path + 'images/statistics.png')
        