import os
import torch
import torch.optim as optim
from tqdm import tqdm

from torch.autograd import Variable

from network_v0.model import PointModel
from loss_function import KeypointLoss

class Trainer(object):
    def __init__(self, config, train_loader=None):
        self.config = config
        # data parameters
        self.train_loader = train_loader
        self.num_train = len(self.train_loader)

        # training parameters
        self.max_epoch = config.max_epoch
        self.start_epoch = config.start_epoch
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.lr_factor = config.lr_factor
        self.display = config.display

        # misc params
        self.use_gpu = config.use_gpu
        self.random_seed = config.seed
        self.gpu = config.gpu
        self.ckpt_dir = config.ckpt_dir
        self.ckpt_name = '{}-{}'.format(config.ckpt_name, config.seed)
		
        # build model
        self.model = PointModel(is_test=False)
        
        # training on GPU
        if self.use_gpu:
            torch.cuda.set_device(self.gpu)
            self.model.cuda()

        print('Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.parameters()])))	
        
        # build loss functional
        self.loss_func = KeypointLoss(config)
        
        # build optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 8], gamma=self.lr_factor)

        # resume
        if int(self.config.start_epoch) > 0:
            self.config.start_epoch, self.model, self.optimizer, self.lr_scheduler = self.load_checkpoint(int(self.config.start_epoch), self.model, self.optimizer, self.lr_scheduler)    
    
    def train(self):
        print("\nTrain on {} samples".format(self.num_train))
        self.save_checkpoint(0, self.model, self.optimizer, self.lr_scheduler)
        for epoch in range(self.start_epoch, self.max_epoch):
            print("\nEpoch: {}/{} --lr: {:.6f}".format(epoch+1, self.max_epoch, self.lr))
            # train for one epoch
            self.train_one_epoch(epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.save_checkpoint(epoch+1, self.model, self.optimizer, self.lr_scheduler)
            
    def train_one_epoch(self, epoch):
        self.model.train()
        for (i, data) in enumerate(tqdm(self.train_loader)):

            if self.use_gpu:
                source_img = data['image_aug'].cuda()
                target_img = data['image'].cuda()
                homography = data['homography'].cuda()
            
            source_img = Variable(source_img)
            target_img = Variable(target_img)
            homography = Variable(homography)
            
            # forward propogation
            output = self.model(source_img, target_img, homography)
            
            # compute loss
            loss, loc_loss, desc_loss, score_loss, corres_loss = self.loss_func(output)

            # compute gradients and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print training info
            msg_batch = "Epoch:{} Iter:{} lr:{:.4f} "\
                        "loc_loss={:.4f} desc_loss={:.4f} score_loss={:.4f} corres_loss={:.4f} "\
                        "loss={:.4f} "\
                        .format((epoch + 1), i, self.lr, loc_loss.data, desc_loss.data, score_loss.data, corres_loss.data, loss.data)

            if((i % self.display) == 0):
                print(msg_batch)
        return

    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        torch.save(
            {'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()},
            os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        ckpt = torch.load(os.path.join(self.ckpt_dir, filename))
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt['epoch']))

        return epoch, model, optimizer, lr_scheduler				        
        
        
        
        
        
        
        
        
        
        
        