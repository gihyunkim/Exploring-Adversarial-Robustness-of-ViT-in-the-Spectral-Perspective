import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
import seaborn as sns

class Utils:
    @staticmethod
    def my_range(start, end, step, mode="sum"):
        r = start
        while r <= end:
            yield r
            if mode=="sum":
                r += step
            elif mode=="mul":
                r *= step

    @staticmethod
    # x if cond is true else y for each index
    def my_where(cond, x, y):
        cond = cond.float()
        return (cond*x) + ((1-cond)*y)

    @staticmethod
    def psnr(y_pred, y_true):
        mse = torch.mean((y_pred-y_true)**2)
        return 20 * torch.log10(1/torch.sqrt(mse))

    @staticmethod
    def show_tensor(tensor, save=False, index=0):
        x = tensor.permute((0, 2, 3, 1))
        if tensor.shape[0] != 0: # if has batch, show only first image
            x = x[0]
        x = torch.squeeze(x, axis=0)
        x = x.to('cpu').detach().numpy()
        if x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if save:
            cv2.imwrite("./%d.jpg"%(index), x*255.0)
        cv2.imshow("tensor", x)
        cv2.waitKey()

    @staticmethod
    def my_sum(original, perturb):
        return original + perturb

    @staticmethod
    def my_mul(original, perturb):
        return original * perturb

    @classmethod
    def get_pass_filter(cls, h, w, r_l=0.4, r_h=np.sqrt(2)):
        center = (int(w/2), int(h/2))
        Y, X = np.ogrid[:h, :w]
        R_l = (h/2) * r_l
        R_h = (h/2) * r_h
        dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
        lp_filter = np.logical_or(dist_from_center <= R_l, dist_from_center > R_h)
        hp_filter = np.logical_not(lp_filter)
        return np.expand_dims(lp_filter.astype(np.int8), [0, 1]), np.expand_dims(hp_filter.astype(np.int8), [0, 1])

    @classmethod
    def getFreqDistribution(cls, mag, dst_path, model_name, idx):
        cls.checkDir(dst_path)
        plt.clf()
        plt.xlabel("Frequency region", fontsize=12)
        plt.ylabel("Proportion", fontsize=12)
        index = [i for i in range(0, 11)]
        mag = torch.unsqueeze(torch.mean(mag, dim=1), dim=1)
        h, w = mag.size()[2], mag.size()[3]
        total_freq = torch.sum(mag).item()
        distribution = []
        for i, ratio in enumerate(cls.my_range(0.1, 1.0, 0.1)):
            lp_filter, hp_filter = cls.get_pass_filter(h, w, ratio)
            if i == 0:
                before_filter = lp_filter
                applied_filter = lp_filter
            else:
                applied_filter = lp_filter - before_filter
                before_filter = lp_filter
            filtered_mag = mag * torch.tensor(applied_filter, device="cuda")
            distribution.append(torch.sum(filtered_mag).item()/ total_freq)
        filtered_mag = mag * torch.tensor(hp_filter, device="cuda")
        distribution.append(torch.sum(filtered_mag).item()/ total_freq)
        palette = sns.color_palette("mako_r", 11)
        sns.barplot(x=index, y=distribution, palette=palette)
        plt.savefig(f'{dst_path}/{model_name}_{idx}.png')
        return distribution

    @staticmethod
    def checkDir(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def remove_inf(x):
        df = pd.DataFrame(x)
        no_inf = df.replace([np.inf, -np.inf], np.nan).dropna()
        return no_inf