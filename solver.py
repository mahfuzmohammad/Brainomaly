from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch
import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


class Solver(object):
    """Solver for training and testing Brainomaly."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # All config
        self.config = config

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Neptune parameters
        self.neptune_id = config.neptune_id
        self.neptune_key = config.neptune_key

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

            # log parameters to neptune
            if config.mode == 'train':
                params = {}
                for k, v in vars(config).items():
                    params[f"Param/{k}"] = v
                self.logger.log(params)
            else:
                assert self.neptune_id is not None, "neptune_id is not defined"
                assert self.neptune_key is not None, "neptune_key is not defined"


    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['MedicalData']:
            self.G = Generator(self.g_conv_dim, 0, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, 0, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # if D_path exists, load it
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        if self.neptune_id is not None:
            self.logger = Logger(self.neptune_id)
        else:
            self.logger = Logger()
            self.neptune_id = self.logger.get_id()

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        # save neptune id to file
        assert self.neptune_id is not None
        with open(os.path.join(self.log_dir.split("/")[0], 'neptune_id'), 'w') as f:
            f.write(self.neptune_id)

        # Set data loader.
        if self.dataset in ['MedicalData']:
            data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixedA, x_fixedB = next(data_iter)
        x_fixedA = x_fixedA.to(self.device)
        x_fixedB = x_fixedB.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_realA, x_realB = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_realA, x_realB = next(data_iter)

            x_realA = x_realA.to(self.device)           # Input images.
            x_realB = x_realB.to(self.device)           # Input images.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            _, out_src = self.D(x_realB)
            d_loss_real = - torch.mean(out_src)

            # Compute loss with fake images.
            mask = self.G(x_realA)
            x_fakeB = torch.tanh(x_realA + mask)
            _, out_src2 = self.D(x_fakeB.detach())
            d_loss_fake =torch.mean(out_src2)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_realB.size(0), 1, 1, 1).to(self.device)
            x_hat2 = (alpha * x_realB.data + (1 - alpha) * x_fakeB.data).requires_grad_(True)
            _, out_src2 = self.D(x_hat2)
            d_loss_gp = self.gradient_penalty(out_src2, x_hat2)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                maskOT = self.G(x_realA)
                x_fakeB2 = torch.tanh(x_realA + maskOT)

                _, out_src2 = self.D(x_fakeB2)
                g_loss_fake = - torch.mean(out_src2)

                # Original-to-original domain.
                maskOO = self.G(x_realB)
                x_fakeB3 = torch.tanh(x_realB + maskOO)
                g_loss_id = torch.mean(torch.abs(x_realB - x_fakeB3))

                g_loss = g_loss_fake + self.lambda_id * g_loss_id

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    self.logger.log(loss)
                    self.logger.log({'Train/epoch': i + 1})

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixedA]
                    mask1 = self.G(x_fixedA)

                    mask1_ = mask1 - torch.min(mask1)
                    mask1_ = mask1_ / torch.max(mask1_)
                    mask1_ = mask1_ * 2
                    mask1_ = mask1_ - 1
                    x_fake_list.append(mask1_.repeat(1, 3, 1, 1))
                    x_fake_list.append(torch.tanh(x_fixedA + mask1))

                    x_fake_list.append(x_fixedB)
                    mask2 = self.G(x_fixedB)

                    mask2_ = mask2 - torch.min(mask2)
                    mask2_ = mask2_ / torch.max(mask2_)
                    mask2_ = mask2_ * 2
                    mask2_ = mask2_ - 1
                    x_fake_list.append(mask2_.repeat(1, 3, 1, 1))
                    x_fake_list.append(torch.tanh(x_fixedB + mask2))

                    x_concat = torch.cat(x_fake_list, dim=3)

                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        self.logger.close()

    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
            
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])


    def testAUCInductive(self):
        """Translate images using Brainomaly trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        assert self.neptune_id is not None, "neptune_id is not defined"
        assert self.neptune_key is not None, "neptune_key is not defined"

        from data_loader import get_loader

        gt_d = {}
        meanp_d = {}

        for gtv, modev in enumerate(['hea', 'ano']):
        
            # Set data loader.
            data_loader = get_loader(self.config.image_dir, self.config.image_size, self.config.batch_size,
                                       'TestValidInductive', self.config.mode + modev, self.config.num_workers)
            
            with torch.no_grad():
                for i, (fname, x_realA) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    imgid = fname[0].split('/')[-1].split('__')[0]

                    x_realA = x_realA.to(self.device)

                    gt_d[imgid] = gtv

                    # Translate images.
                    mask = self.G(x_realA)
                    fake = torch.tanh(x_realA + mask)
                    diff = torch.abs(x_realA - fake)
                    diff /= 2.
                    diff = diff.data.cpu().numpy()
                    meanp = list(np.mean(diff, axis=(1,2,3)))

                    if imgid in meanp_d:
                        meanp_d[imgid] += meanp
                    else:
                        meanp_d[imgid] = meanp

        meanp = []
        gt = []
        ks = []

        for k in gt_d.keys():
            ks.append(k)
            gt.append(gt_d[k])
            meanp.append(np.mean(meanp_d[k]))

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]

        print(f"Threshold: {thmean}")
        meanpth = (np.array(meanp)>=thmean)

        dfcsv = pd.DataFrame.from_dict({
            "pid": ks,
            "gt": gt,
            "pred": meanp,
            "pred_th": meanpth.tolist()
        })

        csv_path = os.path.join(self.log_dir.split("/")[0], str(self.test_iters)+"_inductive.csv")
        dfcsv.to_csv(csv_path, index=False)

        print(f"Unique: {np.unique(meanpth)}")
        print(f"Classification report:\n{classification_report(gt, meanpth)}\n")

        fpr, tpr, threshold = roc_curve(gt, meanp)
        tn, fp, fn, tp = confusion_matrix(gt, meanpth).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        meanauc = auc(fpr, tpr)

        print(f"Model Iter {self.test_iters} AUC: {round(meanauc, 2)}, SEN: {sensitivity}, SPEC: {specificity}")

        log_dict = {
            self.neptune_key + '/AUC': meanauc,
            self.neptune_key + '/SEN': sensitivity,
            self.neptune_key + '/SPEC': specificity,
            self.neptune_key + '/Threshold': thmean,
            self.neptune_key + '/TN': tn,
            self.neptune_key + '/FP': fp,
            self.neptune_key + '/FN': fn,
            self.neptune_key + '/TP': tp,
        }

        self.logger.log_with_step(int(self.test_iters), log_dict)
        self.logger.close()


    def testAUCTransductive(self):
        """Translate images using Brainomaly trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        assert self.neptune_id is not None, "neptune_id is not defined"
        assert self.neptune_key is not None, "neptune_key is not defined"

        from data_loader import get_loader

        gt_d = {}
        meanp_d = {}

        for gtv, modev in enumerate(['hea', 'ano']):

            # Set data loader.
            data_loader = get_loader(self.config.image_dir, self.config.image_size, self.config.batch_size,
                                     'TestValidTransductive', self.config.mode + modev, self.config.num_workers)

            with torch.no_grad():
                for i, (fname, x_realA) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    imgid = fname[0].split('/')[-1].split('__')[0]

                    x_realA = x_realA.to(self.device)

                    gt_d[imgid] = gtv

                    # Translate images.
                    mask = self.G(x_realA)
                    fake = torch.tanh(x_realA + mask)
                    diff = torch.abs(x_realA - fake)
                    diff /= 2.
                    diff = diff.data.cpu().numpy()
                    meanp = list(np.mean(diff, axis=(1, 2, 3)))

                    if imgid in meanp_d:
                        meanp_d[imgid] += meanp
                    else:
                        meanp_d[imgid] = meanp

        meanp = []
        gt = []
        ks = []

        for k in gt_d.keys():
            ks.append(k)
            gt.append(gt_d[k])
            meanp.append(np.mean(meanp_d[k]))

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]

        print(f"Threshold: {thmean}")
        meanpth = (np.array(meanp) >= thmean)

        dfcsv = pd.DataFrame.from_dict({
            "pid": ks,
            "gt": gt,
            "pred": meanp,
            "pred_th": meanpth.tolist()
        })

        csv_path = os.path.join(self.log_dir.split("/")[0], str(self.test_iters) + "_transductive.csv")
        dfcsv.to_csv(csv_path, index=False)

        print(f"Unique: {np.unique(meanpth)}")
        print(f"Classification report:\n{classification_report(gt, meanpth)}\n")

        fpr, tpr, threshold = roc_curve(gt, meanp)
        tn, fp, fn, tp = confusion_matrix(gt, meanpth).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        meanauc = auc(fpr, tpr)

        print(f"Model Iter {self.test_iters} AUC: {round(meanauc, 2)}, SEN: {sensitivity}, SPEC: {specificity}")

        log_dict = {
            self.neptune_key + '/AUC': meanauc,
            self.neptune_key + '/SEN': sensitivity,
            self.neptune_key + '/SPEC': specificity,
            self.neptune_key + '/Threshold': thmean,
            self.neptune_key + '/TN': tn,
            self.neptune_key + '/FP': fp,
            self.neptune_key + '/FN': fn,
            self.neptune_key + '/TP': tp,
        }

        self.logger.log_with_step(int(self.test_iters), log_dict)
        self.logger.close()


    def testAUCp(self):
        """Translate images using Brainomaly trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        assert self.neptune_id is not None, "neptune_id is not defined"
        assert self.neptune_key is not None, "neptune_key is not defined"

        from data_loader import get_loader

        gt_d = {}
        meanp_d = {}

        for gtv, modev in enumerate(['hea', 'ano']):

            # Set data loader.
            data_loader = get_loader(self.config.image_dir, self.config.image_size, self.config.batch_size,
                                     'testAUCp', self.config.mode + modev, self.config.num_workers)

            with torch.no_grad():
                for i, (fname, x_realA) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    imgid = fname[0].split('/')[-1].split('__')[0]

                    x_realA = x_realA.to(self.device)

                    gt_d[imgid] = gtv

                    # Translate images.
                    mask = self.G(x_realA)
                    fake = torch.tanh(x_realA + mask)
                    diff = torch.abs(x_realA - fake)
                    diff /= 2.
                    diff = diff.data.cpu().numpy()
                    meanp = list(np.mean(diff, axis=(1, 2, 3)))

                    if imgid in meanp_d:
                        meanp_d[imgid] += meanp
                    else:
                        meanp_d[imgid] = meanp

        meanp = []
        gt = []
        ks = []

        for k in gt_d.keys():
            ks.append(k)
            gt.append(gt_d[k])
            meanp.append(np.mean(meanp_d[k]))

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]

        print(f"Thresholdp: {thmean}")
        meanpth = (np.array(meanp) >= thmean)

        dfcsv = pd.DataFrame.from_dict({
            "pid": ks,
            "gt": gt,
            "pred": meanp,
            "pred_th": meanpth.tolist()
        })

        csv_path = os.path.join(self.log_dir.split("/")[0], str(self.test_iters) + "_aucp.csv")
        dfcsv.to_csv(csv_path, index=False)

        print(f"Unique: {np.unique(meanpth)}")
        print(f"Classification report:\n{classification_report(gt, meanpth)}\n")

        fpr, tpr, threshold = roc_curve(gt, meanp)
        tn, fp, fn, tp = confusion_matrix(gt, meanpth).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        meanauc = auc(fpr, tpr)

        print(f"Model Iter {self.test_iters} AUCp: {round(meanauc, 2)}, SENp: {sensitivity}, SPECp: {specificity}")

        log_dict = {
            self.neptune_key + '/AUCp': meanauc,
            self.neptune_key + '/SENp': sensitivity,
            self.neptune_key + '/SPECp': specificity,
            self.neptune_key + '/Thresholdp': thmean,
            self.neptune_key + '/TNp': tn,
            self.neptune_key + '/FPp': fp,
            self.neptune_key + '/FNp': fn,
            self.neptune_key + '/TPp': tp,
        }

        self.logger.log_with_step(int(self.test_iters), log_dict)
        self.logger.close()
