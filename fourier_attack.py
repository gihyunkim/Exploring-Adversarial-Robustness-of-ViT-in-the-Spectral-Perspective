import torch
import torch.nn.functional as F
import math
from utils import Utils

class FourierAttack:
    def __init__(self, attack, model, lam, iteration, lr, weight_decay, endure_thres,
                 normalize, invNormalize, device):
        self.attack = attack
        self.model = model
        self.lam = lam
        self.iteration = iteration
        self.lr = lr
        self.weight_decay = weight_decay
        self.endure_thres = endure_thres
        self.normalize = normalize
        self.invNormalize = invNormalize
        self.device = device

    def perturbKeepSymmetric(self, original, perturb, mode="phase"):
        b, c, h, w = original.shape
        perturbed = torch.empty([b, c, h, w], requires_grad=False, device=self.device)
        visualize_perturb = torch.empty([b, c, h, w], requires_grad=False, device=self.device)

        if mode=="phase":
            func = Utils.my_sum
            sine = -1
        elif mode=="mag":
            func = Utils.my_mul
            sine = 1

        '''Keep phase symmetric'''
        mid_p = math.ceil((original.shape[2] - 1) / 2)  # middle index

        # top-left low-frequency
        perturbed[:, :, 0, 0] = func(original[:, :, 0, 0],perturb[:, :, 0, 0])
        visualize_perturb[:, :, 0, 0] = perturb[:, :, 0, 0]

        # left
        perturbed[:, :, 1:mid_p + 1, 0] = func(original[:, :, 1:mid_p + 1, 0], perturb[:, :, 1:mid_p + 1, 0])
        perturbed[:, :, mid_p + 1:, 0] = sine * torch.flip(perturbed[:, :, 1:mid_p, 0], dims=[2])
        visualize_perturb[:, :, 1:mid_p+ 1, 0] = perturb[:, :, 1:mid_p+1, 0]
        visualize_perturb[:, :, mid_p+1:, 0] = sine * torch.flip(perturb[:, :, 1:mid_p, 0], dims=[2])

        # top
        perturbed[:, :, 0, 1:mid_p + 1] = func(original[:, :, 0, 1:mid_p + 1], perturb[:, :, 0, 1:mid_p + 1])
        perturbed[:, :, 0, mid_p + 1:] = sine * torch.flip(perturbed[:, :, 0, 1:mid_p], dims=[2])
        visualize_perturb[:, :, 0, 1:mid_p+1] = perturb[:, :, 0, 1:mid_p+1]
        visualize_perturb[:, :, 0, mid_p+1:] = sine * torch.flip(perturb[:,:,0,1:mid_p], dims=[2])

        # middle
        perturbed[:, :, mid_p, 1:mid_p + 1] = func(original[:, :, mid_p, 1:mid_p + 1] ,perturb[:, :, mid_p, 1:mid_p + 1])
        perturbed[:, :, mid_p, mid_p + 1:] = sine * torch.flip(perturbed[:, :, mid_p, 1:mid_p], dims=[2])
        visualize_perturb[:, :, mid_p, 1:mid_p+1] = perturb[:, :, mid_p, 1:mid_p+1]
        visualize_perturb[:, :, mid_p, mid_p+1:] = sine * torch.flip(perturb[:, :, mid_p, 1:mid_p], dims=[2])

        # top & bottom middle
        perturbed[:, :, 1:mid_p, 1:] = func(original[:, :, 1:mid_p, 1:] ,perturb[:, :, 1:mid_p, 1:])
        perturbed[:, :, mid_p + 1:, 1:] = sine * torch.flip(perturbed[:, :, 1:mid_p, 1:], dims=[2, 3])
        visualize_perturb[:, :, 1:mid_p, 1:] = perturb[:, :, 1:mid_p, 1:]
        visualize_perturb[:, :, mid_p+1:, 1:] = sine * torch.flip(perturb[:, :, 1:mid_p, 1:], dims=[2, 3])
        return perturbed, visualize_perturb

    def retrieveImg(self, phase, spectrum):
        complex_elem = torch.complex(torch.cos(phase),
                                     torch.sin(phase))
        retrieved_img = spectrum * complex_elem
        retrieved_img = torch.fft.ifft2(retrieved_img)
        retrieved_img = retrieved_img.real
        return retrieved_img

    def fourierAttack(self, batch_x, batch_y):
        self.model.eval()
        lam = self.lam
        b, c, h, w = batch_x.shape
        found = False

        '''perturbation'''
        perturb_phase = torch.zeros([b, c, h, w], device=self.device)
        perturb_mag = torch.ones([b, c, h, w],  device=self.device)
        perturb_pixel = torch.zeros([b, c, h, w], device=self.device)

        optim = torch.optim.Adam([perturb_phase, perturb_mag, perturb_pixel], lr=self.lr, weight_decay=self.weight_decay)

        if self.attack == "phase":
            perturb_phase.requires_grad_(True)
        elif self.attack == "mag":
            perturb_mag.requires_grad_(True)
        elif self.attack == 'pixel':
            perturb_pixel.requires_grad_(True)
        elif self.attack == 'phase+mag':
            perturb_phase.requires_grad_(True)
            perturb_mag.requires_grad_(True)
        elif self.attack == 'all':
            perturb_phase.requires_grad_(True)
            perturb_mag.requires_grad_(True)
            perturb_pixel.requires_grad_(True)

        best_loss = 1e+10
        endure_cnt = 0

        ''' fourier transform'''
        f = torch.fft.fft2(batch_x)
        real_part, imag_part = f.real, f.imag
        phase = torch.angle(f)
        mag = torch.sqrt(imag_part ** 2 + real_part ** 2)

        for i in range(self.iteration):
            '''only do when optimizing phase'''
            if not torch.all(perturb_phase == 0):
                perturbed_phase, visualize_perturb = self.perturbKeepSymmetric(phase, perturb_phase, mode="phase")
            else:
                perturbed_phase = phase + perturb_phase

            '''only do when optimizing magnitude'''
            if not torch.all(perturb_mag == 1):
                perturbed_mag, visualize_perturb_mag = self.perturbKeepSymmetric(mag, perturb_mag, mode="mag")
                perturbed_mag = torch.maximum(torch.full_like(perturbed_mag, 1e-6),
                                              perturbed_mag)  # magnitude must be positive
            else:
                perturbed_mag = mag * perturb_mag

            adv_img = self.retrieveImg(perturbed_phase, perturbed_mag) + perturb_pixel
            adv_img = torch.clamp(adv_img, 0, 1)

            '''loss'''
            pred = self.model(self.normalize(adv_img))
            pred_adv = torch.argmax(pred, 1)

            mse_loss = F.mse_loss(adv_img, batch_x)
            ce_loss = -F.cross_entropy(pred, batch_y)
            loss = lam * mse_loss + ce_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            '''whether found adversarial example'''
            if not torch.equal(pred_adv, batch_y):
                found = True

            if loss < best_loss:
                endure_cnt = 0
            else:
                endure_cnt += 1

            if endure_cnt > self.endure_thres:
                break

        return adv_img, found