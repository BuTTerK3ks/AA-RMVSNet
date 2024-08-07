import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
import ast
from collections import OrderedDict

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--return_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=960, help='Maximum image width when training.')
parser.add_argument('--image_scale', type=float, default=1.0, help='pred depth map scale (compared to input image)') 

parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=7, help='training view num setting')

parser.add_argument('--dataset', default='data_eval_transform', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=256, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.0, help='the depth interval scale')


parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')

# parse arguments and check
args = parser.parse_args()
print_args(args)

model_name = str.split(args.loadckpt, '/')[-2] + '_' + str.split(args.loadckpt, '/')[-1]
save_dir = os.path.join(args.outdir, model_name)
if not os.path.exists(save_dir):
    print('save dir', save_dir)
    os.makedirs(save_dir)

# run MVS model to save depth maps and confidence maps
def save_depth():
    
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth,
                    adaptive_scaling=True, max_h=args.max_h, max_w=args.max_w, sample_scale=1, base_image_size=8)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = EMVSNet(disparity_level=32, image_scale=args.image_scale,
                    max_h=args.max_h, max_w=args.max_w, return_depth=args.return_depth)


    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))

    '''

    # Allow both keys xxx & module.xxx in dict
    state_dict = torch.load(args.loadckpt)
    if "module.feature.conv0_0.0.weight" in state_dict['model']:
        print("With module in keys")
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict['model'],True)
        
    else:
        print("No module in keys")
        model.load_state_dict(state_dict['model'], True)
        model = nn.DataParallel(model)
        
    '''
    # Load the checkpoint
    state_dict = torch.load(args.loadckpt)['model']  # Assuming 'model' is the key under which the state dict is saved

    # Create a new state dictionary without the 'module.' prefix

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' of each key
        new_state_dict[name] = v

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict, strict=True)

    model.cuda()
    model.eval()
    
    count = -1
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            count += 1
            try:
                print('process', sample['filename'])
                sample_cuda = tocuda(sample)
                print('input shape: ', sample_cuda["imgs"].shape, sample_cuda["proj_matrices"].shape, sample_cuda["depth_values"].shape)
                time_s = time.time()
                outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

                one_time = time.time() - time_s
                total_time += one_time
                print('one forward: ', one_time)
                if count % 50 == 0:
                    print('avg time:', total_time / 50)
                    total_time = 0

                outputs = tensor2numpy(outputs)
                del sample_cuda
                print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
                filenames = sample["filename"]
                outputs = [outputs]

                # save depth maps and confidence maps
                for filename, output in zip(filenames, outputs):
                    depth_filename_pfm = os.path.join(save_dir, filename.format('depth_est_{}'.format(0), '.pfm'))
                    confidence_filename_pfm = os.path.join(save_dir, filename.format('confidence_{}'.format(0), '.pfm'))
                    epistemic_filename_pfm = os.path.join(save_dir, filename.format('epistemic_{}'.format(0), '.pfm'))
                    aleatoric_filename_pfm = os.path.join(save_dir, filename.format('aleatoric_{}'.format(0), '.pfm'))



                    depth_filename_png = os.path.join(save_dir, filename.format('depth_png_{}'.format(0), '.png'))
                    aleatoric_filename_png = os.path.join(save_dir, filename.format('aleatoric_{}'.format(0), '.png'))
                    epistemic_filename_png = os.path.join(save_dir, filename.format('epistemic_{}'.format(0), '.png'))

                    os.makedirs(depth_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(depth_filename_png.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(aleatoric_filename_png.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(epistemic_filename_png.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(epistemic_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(aleatoric_filename_pfm.rsplit('/', 1)[0], exist_ok=True)

                    depth_est = output["depth"]
                    photometric_confidence = output["photometric_confidence"]
                    evidential_prediction = output["evidential_prediction"]

                    gamma, nu, alpha, beta = evidential_prediction[0, :, :], evidential_prediction[1, :, :], evidential_prediction[2, :, :], evidential_prediction[3, :, :]

                    aleatoric_1 = np.sqrt(beta * (nu + 1) / nu / alpha)
                    epistemic_1 = 1. / np.sqrt(nu)

                    save_png(gamma, depth_filename_png, title="Estimated Depth", mode="depth")
                    save_png(aleatoric_1, aleatoric_filename_png, title="Aleatoric uncertainty")
                    save_png(epistemic_1, epistemic_filename_png, title="Epistemic uncertainty")



                    # save depth maps
                    save_pfm(depth_filename_pfm, gamma)
                    # save confidence maps
                    save_pfm(confidence_filename_pfm, photometric_confidence.squeeze())
                    save_pfm(epistemic_filename_pfm, epistemic_1)
                    save_pfm(aleatoric_filename_pfm, aleatoric_1)
            except:
                print("PROBLEM!!!")






if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    print('save depth *******************\n')
    save_depth()
