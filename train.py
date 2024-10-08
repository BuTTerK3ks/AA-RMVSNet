import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import islice

import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import sys
import datetime
import ast
from datasets.data_io import *

from evidential.models import *
from evidential.save import *
from statistics import *

from torchsummary import summary

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Codebase for AA-RMVSNet')
parser.add_argument('--mode', default='train', help='train, val or test')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--save_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width when training.')

parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=3, help='training view num setting')

parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='val list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--save_dir', default=None, help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq_checkpoint', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("current time", current_time_str)

print("creating new summary file")
logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

SAVE_DEPTH = args.save_depth
if SAVE_DEPTH:
    if args.save_dir is None:
        sub_dir, ckpt_name = os.path.split(args.loadckpt)
        index = ckpt_name[6:-5]
        save_dir = os.path.join(sub_dir, index)
    else:
        save_dir = args.save_dir
    print(os.path.exists(save_dir), ' exists', save_dir)
    if not os.path.exists(save_dir):
        print('save dir', save_dir)
        os.makedirs(save_dir)


MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, -1, args.image_scale) # Training with False, Test with inverse_depth
#val_dataset = MVSDataset(args.trainpath, args.vallist, "val", 5, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) #view_num = 5, light_idx = 3
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) # use 3
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=12, drop_last=True, prefetch_factor=5)
#ValImgLoader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False, prefetch_factor=5)
# Use test set (with gt depths) for validation

print('Model: EMVSNet')
model = EMVSNet(disparity_level=args.numdepth, image_scale=args.image_scale, max_h=args.max_h, max_w=args.max_w)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total Parameters: {total_params:,}')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training Parameters: {total_trainable_params:,}')

'''
if args.loadckpt:

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))

    # Allow both keys xxx & module.xxx in dict
    state_dict = torch.load(args.loadckpt)
    if "module.feature.conv0_0.0.weight" in state_dict['model']:
        print("With module in keys")
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict['model'], True)

    else:
        print("No module in keys")
        model.load_state_dict(state_dict['model'], True)
        model = nn.DataParallel(model)


model = model.cuda()
model = nn.parallel.DataParallel(model)
'''



if args.loadckpt:
    # Load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))

    # Load the state dictionary
    state_dict = torch.load(args.loadckpt)

    # Check if keys have "module." prefix and remove it if necessary
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v

    # Load state dictionary into the model
    model.load_state_dict(new_state_dict, strict=True)

    # Wrap the model with DataParallel if necessary
    model = nn.DataParallel(model)

# Move model to GPU
model = model.cuda()


print('Optimizer: Adam \n')
optimizer = optim.Adam(model.parameters(), lr=args.lr)



# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming from:", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    print(optimizer)

    start_epoch = state_dict['epoch'] + 1

elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


# main function
def train():
    print('run train()')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-06)
    ## get intermediate learning rate
    for _ in range(start_epoch):
        lr_scheduler.step()
    for epoch_idx in range(start_epoch, args.epochs):

        
        print('Epoch {}/{}:'.format(epoch_idx, args.epochs))

        global_step = len(TrainImgLoader) * epoch_idx
        print('Start Training')
        # training
        #TODO Hier wird nur bis x trainiert
        for batch_idx, sample in enumerate(TrainImgLoader):
        #for batch_idx, sample in enumerate(islice(TrainImgLoader, 0, 100, 1)):
            try:
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = (global_step % args.summary_freq == 0)
                loss, scalar_outputs, image_outputs, evidential_outputs = train_sample(sample, detailed_summary=do_summary)

                for param_group in optimizer.param_groups:
                    lr = param_group['lr']

                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    logger.add_scalar('train/lr', lr, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    save_pytorch(args.logdir, 'train', global_step, image_outputs, evidential_outputs)
                del scalar_outputs, image_outputs
                print(
                    'Epoch {}/{}, Iter {}/{}, LR {}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                         len(TrainImgLoader), lr, loss,
                                                                                         time.time() - start_time))

            except:
                print("Problem with: " + str(batch_idx))


        lr_scheduler.step()
        # checkpoint
        if (epoch_idx + 1) % args.save_freq_checkpoint == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        avg_test_scalars = DictAverageMeter()
        # TODO Hier wird nur bis x getestet

        for batch_idx, sample in enumerate(TestImgLoader):
        #for batch_idx, sample in enumerate(islice(TestImgLoader, 0, 100, 1)):
            start_time = time.time()
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs, evidential_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
                save_pytorch(args.logdir, 'test', global_step, image_outputs, evidential_outputs)
            avg_test_scalars.update(scalar_outputs)
            
            print('Epoch: {}/{}, Iter: {}/{}, Views: {}, test loss = {:.3f}, time = {:3f}, ame = {:3f}, thres2mm = {:3f}, thres4mm = {:3f}, thres8mm = {:3f}, thres16mm = {:3f}, thres32mm = {:3f}'.format(
                                epoch_idx, args.epochs, batch_idx,
                                len(TestImgLoader), args.view_num, loss,
                                time.time() - start_time,
                                scalar_outputs["abs_depth_error"], scalar_outputs["thres2mm_error"], 
                                scalar_outputs["thres4mm_error"], scalar_outputs["thres8mm_error"],
                                scalar_outputs["thres16mm_error"], scalar_outputs["thres32mm_error"]))

            del image_outputs

        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())


def train_sample(sample, detailed_summary=False):

    model.train()
    optimizer.zero_grad()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    probability_volume, evidential, probabilities = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    outputs = {
        "probability_volume": probability_volume,
        'evidential_prediction': evidential
    }

    loss, depth_est, evidential_outputs = loss_der(outputs, depth_gt, mask, depth_value)

    loss.backward()
    optimizer.step()

    std_dev = std_prob(probabilities)
    aleatoric_1_by_total, epistemic_1_by_total, aleatoric_2_by_total, epistemic_2_by_total = divide_by_total(evidential_outputs)
    error_map = (depth_est - depth_gt).abs() * mask

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "std_dev": std_dev,
                     "mask": sample["mask"],
                     "alea_1": evidential_outputs["aleatoric_1"],
                     "epis_1": evidential_outputs["epistemic_1"],
                     "alea_2": evidential_outputs["aleatoric_2"],
                     "epis_2": evidential_outputs["epistemic_2"],
                     "aleatoric_1_by_total": aleatoric_1_by_total,
                     "epistemic_1_by_total": epistemic_1_by_total,
                     "aleatoric_2_by_total": aleatoric_2_by_total,
                     "epistemic_2_by_total": epistemic_2_by_total,
                     "error_map": error_map,
                     }

    if detailed_summary:
        scalar_outputs["aleatoric_1"] = torch.mean(evidential_outputs["aleatoric_1"]).item()
        scalar_outputs["epistemic_1"] = torch.mean(evidential_outputs["epistemic_1"]).item()
        scalar_outputs["aleatoric_2"] = torch.mean(evidential_outputs["aleatoric_2"]).item()
        scalar_outputs["epistemic_2"] = torch.mean(evidential_outputs["epistemic_2"]).item()
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
        scalar_outputs["thres16mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 16)
        scalar_outputs["thres32mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 32)

    # clear cache
    torch.cuda.empty_cache()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, evidential_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    probability_volume, evidential, probabilities = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    outputs = {
        "probability_volume": probability_volume,
        'evidential_prediction': evidential
    }

    prob_volume = outputs['probability_volume']
    loss, depth_est, evidential_outputs = loss_der(outputs, depth_gt, mask, depth_value)

    std_dev = std_prob(probabilities)
    aleatoric_1_by_total, epistemic_1_by_total, aleatoric_2_by_total, epistemic_2_by_total = divide_by_total(evidential_outputs)
    error_map = (depth_est - depth_gt).abs() * mask

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "std_dev": std_dev,
                     "mask": sample["mask"],
                     "alea_1": evidential_outputs["aleatoric_1"],
                     "epis_1": evidential_outputs["epistemic_1"],
                     "alea_2": evidential_outputs["aleatoric_2"],
                     "epis_2": evidential_outputs["epistemic_2"],
                     "aleatoric_1_by_total": aleatoric_1_by_total,
                     "epistemic_1_by_total": epistemic_1_by_total,
                     "aleatoric_2_by_total": aleatoric_2_by_total,
                     "epistemic_2_by_total": epistemic_2_by_total,
                     "error_map": error_map,
                     }

    scalar_outputs["aleatoric_1"] = torch.mean(evidential_outputs["aleatoric_1"]).item()
    scalar_outputs["epistemic_1"] = torch.mean(evidential_outputs["epistemic_1"]).item()
    scalar_outputs["aleatoric_2"] = torch.mean(evidential_outputs["aleatoric_2"]).item()
    scalar_outputs["epistemic_2"] = torch.mean(evidential_outputs["epistemic_2"]).item()
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["thres16mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 16)
    scalar_outputs["thres32mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 32)

    # clear cache
    torch.cuda.empty_cache()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, evidential_outputs

if __name__ == '__main__':
    train()