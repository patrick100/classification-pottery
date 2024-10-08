import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
from confusion_matrix import plot_confusion_matrix

class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda('cuda:0')
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):
        #manual seed
        manualSeed = 0
        np.random.seed(manualSeed)

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                
                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda('cuda:0')
                else:
                    in_data = Variable(data[1].cuda('cuda:0'))
                target = Variable(data[0]).cuda('cuda:0').long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                
                loss = self.loss_fn(out_data, target)
                
                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.5f; train_acc %.5f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc,y_test,y_pred = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            #uncomment to save confusion matrices 
            #if(epoch>=39):
            #    class_names = np.array(['Alabastron','Amphora', 'Hydria', 'Kalathos', 'Krater', 'Kylix','Lekythos','Native-American','Pelike','Picher-Shaped','Psykter'])
            #    np.savez('max_1/data_matrix_confusion_max_1'+'_'+str(epoch+1)+'.npz', y_test=y_test, y_pred=y_pred,val_overall_acc=val_overall_acc)
            #    plot_confusion_matrix(str(epoch+1),y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix '+str(val_overall_acc)+'%')

            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(11)
        samples_class = np.zeros(11)
        all_loss = 0
        
        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []
        #only necessary to generate the matrix confusion
        y_test = []
        y_pred = []
        ###
        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda('cuda:0')
            else:#'svcnn'
                in_data = Variable(data[1]).cuda('cuda:0')
            target = Variable(data[0]).cuda('cuda:0')

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]

            y_test = np.concatenate([y_test, target.cpu().data.numpy().astype('int')])
            y_pred = np.concatenate([y_pred, pred.cpu().data.numpy().astype('int')])

            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)
        
        self.model.train()

        y_test = y_test.astype('int')
        y_pred = y_pred.astype('int')
        
        return loss, val_overall_acc, val_mean_class_acc, y_test, y_pred