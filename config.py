import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='LANet')

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# train data params
traindata_arg = add_argument_group('Traindata Params')
traindata_arg.add_argument('--train_txt', type=str, default='',
                            help='Train set.')
traindata_arg.add_argument('--train_root', type=str, default='',
                            help='Where the train images are.')
traindata_arg.add_argument('--batch_size', type=int, default=8,
                            help='# of images in each batch of data')
traindata_arg.add_argument('--num_workers', type=int, default=4,
                            help='# of subprocesses to use for data loading')
traindata_arg.add_argument('--pin_memory', type=str2bool, default=True,
                            help='# of subprocesses to use for data loading')
traindata_arg.add_argument('--shuffle', type=str2bool, default=True,
                            help='Whether to shuffle the train and valid indices')
traindata_arg.add_argument('--image_shape', type=tuple, default=(240, 320),
                            help='')
traindata_arg.add_argument('--jittering', type=tuple, default=(0.5, 0.5, 0.2, 0.05),
                            help='')

# data storage
storage_arg = add_argument_group('Storage')
storage_arg.add_argument('--ckpt_name', type=str, default='PointModel',
                            help='')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--start_epoch', type=int, default=0,
                        help='')
train_arg.add_argument('--max_epoch', type=int, default=12,
                        help='')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                        help='Initial learning rate value.')
train_arg.add_argument('--lr_factor', type=float, default=0.5,
                        help='Reduce learning rate value.')	
train_arg.add_argument('--momentum', type=float, default=0.9,
                        help='Nesterov momentum value.')			   
train_arg.add_argument('--display', type=int, default=50,
                        help='')

# loss function params
loss_arg = add_argument_group('Loss function Params')
loss_arg.add_argument('--score_weight', type=float, default=1.,
                        help='')
loss_arg.add_argument('--loc_weight', type=float, default=1.,
                        help='')
loss_arg.add_argument('--desc_weight', type=float, default=4.,
                        help='')
loss_arg.add_argument('--corres_weight', type=float, default=.5,
                        help='')
loss_arg.add_argument('--corres_threshold', type=int, default=4.,
                        help='')
					   
# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                        help="Whether to run on the GPU.")
misc_arg.add_argument('--gpu', type=int, default=0,
                        help="Which GPU to run on.")										  
misc_arg.add_argument('--seed', type=int, default=1001,
                        help='Seed to ensure reproducibility.')					  
misc_arg.add_argument('--ckpt_dir', type=str, default='./checkpoints',
                        help='Directory in which to save model checkpoints.')					  

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
