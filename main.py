from torch.utils.data import DataLoader
import load_datasets
import os
import numpy as np
import torch
from utils import Utils
from torchvision.utils import save_image
from get_model import GetModel
from fourier_attack import FourierAttack
from piq import multi_scale_ssim, LPIPS, mdsi
import random
import argparse
import pandas as pd
import tqdm

'''
    Possible models
    ['resnet50', 'resnet152', 'vit-b-1k', 'vit-b-21k', 
    'vit-l', 'swin-b', 'deit-s', 'deit-s-nodist']
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', required=True,
                        default='resnet50', help='name of model, you might want to refer get_model.py')
    parser.add_argument('--attack', required=True, default='phase',
                        help='Attacks in Fourier attack framework, [\'phase\', \'mag\', \'pixel\','
                             ' \'phase+mag\', \'all\']')
    parser.add_argument('--lam', default=5e+4, type=float, help='weight parameter to MSE Loss')
    parser.add_argument('--iteration', default=1000, type=int, help='max iteration for finding adversarial examples')
    parser.add_argument('--lr',default=5e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-6, type=float, help='weight decay')
    parser.add_argument('--endure_thres', default=5, type=int, help='enduring count for no loss decrease')
    parser.add_argument('--save',  action='store_true', help='save image and distribution')
    parser.add_argument('--src_path', default='./samples', help='root dir for dataset')
    parser.add_argument('--label_path', default='./label.txt', help='path for label file')
    parser.add_argument('--dst_path', default='./saved', help='dir for saving results')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    args = parser.parse_args(["--model_name", "vit-b-21k", "--attack", "phase", '--save'])
    return args

class FourierAdversarialAttack:
    def __init__(self, args):
        '''seed'''
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        '''Dataset Setting'''
        image_csv = "./data/images.csv"
        image_root = "./data/images/"
        image_info = pd.read_csv(image_csv)
        images = image_info["ImageId"]
        labels = image_info["TrueLabel"]
        self.img_paths = [os.path.join(image_root, images[idx] + ".png") for idx in range(len(images))]

        '''make dir for dst path'''
        Utils.checkDir(args.dst_path)

        self.img_shape = (224, 224)

        '''setup'''
        self.incorrect = 0
        self.save_idx = 0

        '''Model'''
        self.model, self.transform, self.normalize, self.invNormalize = GetModel.getModel(args.model_name, pretrained=True)
        self.model.to(device)

        '''attacker'''
        self.attacker = FourierAttack(attack=args.attack, model=self.model, lam=args.lam, iteration=args.iteration,
                                      lr=args.lr, weight_decay=args.decay, endure_thres=args.endure_thres,
                                      normalize=self.normalize, invNormalize=self.invNormalize, device=device)
        '''Data Load'''
        self.datasets = load_datasets.AdversarialTrainDataset(self.img_paths, labels, self.transform)
        self.loader = DataLoader(self.datasets, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    '''Predict code for model'''
    def predict(self):
        self.model.eval()
        total = len(self.datasets)
        correct = 0
        with torch.no_grad():
            for images, labels in self.loader:
                output = self.model(self.normalize(images))
                predicted = torch.argmax(output, axis=1)
                correct += (predicted == labels).sum().item()
        print("accuracy: %f" % (correct / total))

    def save_img(self, img, dst_path, mode, model_name, idx=0, inv_norm=None):
        dst_path = os.path.join(dst_path, mode)
        Utils.checkDir(dst_path)
        if model_name:
            dst_path = os.path.join(dst_path, model_name)
            Utils.checkDir(dst_path)
        if inv_norm:
            img = inv_norm(img)
            img = torch.clamp(img, 0, 1)
        '''save image'''
        save_image(img, "%s/%s_%d.png"%(dst_path, mode, idx), nrow=4)

    def attack(self, dst_path, model_name, lam, save):
        psnr_list, ssim_list, acc_list = [], [], []
        save_idx = 1
        check_idx = 0

        print('Start find adversarial examples')
        for i, (batch_x, batch_y) in enumerate(tqdm.tqdm(self.loader)):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            '''Attack'''
            adv_img, found = self.attacker.fourierAttack(batch_x, batch_y)

            '''found adversarial Examples'''
            if found:
                self.incorrect += 1

            '''Quality metrics'''
            psnr_ratio, ssim_ratio = torch.mean(Utils.psnr(batch_x, adv_img), dim=0).item(), torch.mean(multi_scale_ssim(batch_x, adv_img), dim=0).item()
            ssim_list.append(ssim_ratio)
            psnr_list.append(psnr_ratio)
            print(f"Idx_{i}: adv_exam_found: {str(found)}, PSNR/SSIM compared to clean img: {psnr_ratio}/{ssim_ratio}")

            '''for save pertubation img'''
            if save and found:
                '''forier transform for drawing distribution'''
                perturb = torch.subtract(adv_img, batch_x)
                perturb_fourier = torch.fft.fftshift(torch.fft.fft2(perturb))
                real_part, imag_part = perturb_fourier.real, perturb_fourier.imag
                perturb_mag = torch.sqrt(imag_part ** 2 + real_part ** 2)
                Utils.getFreqDistribution(perturb_mag, dst_path=os.path.join(dst_path, "hist"), model_name=model_name, idx=save_idx)

                self.save_img(batch_x, dst_path ,"origin", model_name=None, idx=save_idx,inv_norm=None) # original image
                self.save_img(adv_img, dst_path, "adv", model_name=model_name, idx=save_idx, inv_norm=None) # attack image
                save_perturb = torch.clamp(20*torch.abs(perturb), 0, 1)
                self.save_img(save_perturb, dst_path, "perturb", model_name=model_name,idx=save_idx, inv_norm=None) # perturbation image
                save_idx+=1
            check_idx += 1
        total_accuracy = 1 - (self.incorrect / len(self.loader))

        '''remove wrong calculate data'''
        ssim_list = Utils.remove_inf(ssim_list)
        psnr_list = Utils.remove_inf(psnr_list)

        total_ssim = np.array(ssim_list).mean()
        total_psnr = np.array(psnr_list).mean()

        print("model: %s, acc: %.3f,  lam: %f, psnr: %f, ssim: %f" %(model_name, total_accuracy, lam, total_psnr, total_ssim))
        return total_accuracy, total_psnr, total_ssim
def main():
    args = get_config()
    f_attack = FourierAdversarialAttack(args)
    f_attack.attack(args.dst_path, args.model_name, args.lam, args.save)

if __name__ == "__main__":
    main()


