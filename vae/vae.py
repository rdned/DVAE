import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceGraph_ELBO
from pyro.optim import AdamW

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
import scipy.sparse as ss

import numpy as np
import math
import time

from vae.utils.utils import asMinutes
from vae.utils.logger import logger
from vae.utils.custom_mlp import MLP
from vae.config.hyperparameters import HIDDEN_DIM, Z_DIM, MINIBATCH_SIZE, SUBSAMPLE_RATIO, LR_ADAM, BETAS, ALPHA1, ALPHA2, BETA, BCP

print(f"Is cuda available? {torch.cuda.is_available()}")
print(f"Cuda device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Cuda device name: {torch.cuda.get_device_name()}") # with an argument? torch.cuda.get_device_name(0)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
USE_CUDA = True if torch.cuda.is_available() else False

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

TH_PRED = 0.5  # threshold for binarizing the predicion: [0,1] --> {0,1}

print(f"Pyro version: {pyro.__version__}")
#assert pyro.__version__.startswith('1.7.0')
pyro.enable_validation(True)
pyro.set_rng_seed(101)


class Z_loc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.Softmax(dim=-1)(x)


class Z_scale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Y_loc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loc_y = torch.tanh(x) / 2 + .5
        loc_y = loc_y.squeeze(-1)
        return loc_y


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self, alpha1=ALPHA1, alpha2=ALPHA2, beta=BETA, corruption=BCP, z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, use_cuda=USE_CUDA, **data_tr):
        super().__init__()
        logger.debug(f"Pyro: {pyro.__version__}")

        self.alpha1=alpha1
        self.alpha2=alpha2
        self.beta=beta
        self.corruption=corruption
        self.z_dim = z_dim
        n_words = data_tr['encoder'].shape[1]
        self.n_words = n_words
        features_dict = {
            k: torch.Tensor(v.toarray() if isinstance(v, ss.spmatrix) else v )
            for (k, v) in data_tr.items() if k not in ['y']
        }
        self.mean_std = {k: [features_dict[k].mean(0), features_dict[k].std(0)] for k in features_dict.keys()}
        self.encoder = MLP(
            name = 'encoder',
            mlp_sizes = [n_words] + [8*hidden_dim] + [[z_dim, z_dim]],
            activation=nn.Softplus,
            output_activation=[Z_loc, Z_scale],
            post_act_fct=lambda layer_ix, total_layers, layer: None,
            allow_broadcast=True,
            use_cuda=use_cuda,
        )

        self.decoder = MLP(
            name = "decoder",
            mlp_sizes= [z_dim] + [hidden_dim//2, hidden_dim//4] + [1],
            activation=nn.Softplus,
            output_activation=Y_loc,
            post_act_fct=lambda layer_ix, total_layers, layer: None,
            allow_broadcast=True,
            use_cuda=use_cuda,
        )
        self.decoder_x = MLP(
            name = "decoder_x",
            mlp_sizes = [z_dim] + [hidden_dim] + [n_words],
            activation=nn.Softplus,
            output_activation=None,
            post_act_fct=lambda layer_ix, total_layers, layer: None,
            allow_broadcast=True,
            use_cuda=use_cuda,
        )

        if use_cuda:
            # calling cuda() here will put all the parameters of the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda

    # define the model p(x|z)p(z)
    # @config_enumerate
    def model(self, batch_dict, sample_size):
        pyro.module(self.decoder.name, self.decoder)
        pyro.module(self.decoder_x.name, self.decoder_x)
        data_axis = pyro.plate("data", sample_size)
        z_loc = torch.zeros(self.z_dim, device=device)
        with data_axis as ind:  # pass a device argument to plate if data is on the GPU
            with pyro.poutine.scale(scale=self.beta):
                Z = pyro.sample(
                    f"latent_{self.encoder.name}",
                    dist.MultivariateNormal(z_loc, torch.eye(self.z_dim))
                    )
        loc_y = self.decoder.forward(Z)
        loc_x = self.decoder_x.forward(Z)
        with data_axis as ind:
            with pyro.poutine.scale(scale=self.alpha2):
                pyro.sample("obs",
                        dist.Bernoulli(loc_y),  # .to_event(1),
                        infer={"enumerate": "sequential"},
                        obs=batch_dict['y'].float().index_select(0, ind).detach()
                )
            with pyro.poutine.scale(scale=self.alpha1):
                eye_matrix = torch.eye(self.n_words, device=loc_x.device)
                pyro.sample(
                    "reconstruct_x",
                    dist.MultivariateNormal(loc_x, eye_matrix),
                    obs=batch_dict['encoder'].squeeze().float().index_select(0, ind).detach()
                )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, batch_dict, sample_size):
        baseline_dict = {'use_decaying_avg_baseline': True,
                         'baseline_beta': 0.85}
        pyro.module(self.encoder.name, self.encoder)
        subsample_size = sample_size // SUBSAMPLE_RATIO if sample_size >= 10 * SUBSAMPLE_RATIO else sample_size
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as ind:
            x = batch_dict['corr_'+self.encoder.name]
            if ind is not None:
                x = x.index_select(0, ind)
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample(
                f"latent_{self.encoder.name}",
                dist.Normal(z_loc, z_scale).to_event(1),
                infer=dict(baseline=baseline_dict),
            )

    def classify(self, batch_dict):
        x = batch_dict[self.encoder.name]
        z_loc, z_scale = self.encoder.forward(x)
        return self.decoder.forward(z_loc)

    def reconstruct_x(self, batch_dict):
        x = batch_dict[self.encoder.name]
        z_loc, z_scale = self.encoder.forward(x)
        return self.decoder_x.forward(z_loc), z_loc

    def infer_parameters(self, num_epochs=40, batch_size=MINIBATCH_SIZE, dt_tr=dict()):
        prob_tensor_cache = {} 

        def corrupt(X, p):
            if p > 0:
                dixs, wixs = X.nonzero(as_tuple=True)
                mask = torch.zeros_like(X)
                    
                # Check if we already created the tensor for this device
                if X.device not in prob_tensor_cache:
                    prob_tensor_cache[X.device] = torch.tensor([1 - p], device=X.device)
                    
                mask[dixs, wixs] = dist.Bernoulli(prob_tensor_cache[X.device]).expand([len(dixs)]).sample()
                return X * mask
            else:
                logger.debug("No Bernoulli corruption!!")
                return X
            
        logger.debug("Starting training!")
        start = time.time()
        pyro.clear_param_store()

        train_loader = create_loader(batch_size=batch_size, use_cuda=self.use_cuda, **dt_tr)

        # setup the optimizer
        adam_args = {"lr": LR_ADAM,
                     "betas": BETAS
                     }
        optimizer = AdamW(adam_args)
        elbo = JitTraceGraph_ELBO(
            strict_enumeration_warning=False,
        )

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
        N_data = train_loader.__len__()

        perfect_tr = []
        for epoch in range(num_epochs):
            epoch_loss = 0.
            if epoch > 0:
                pred_train_old = pred_train.copy()
            pred_train = []

            for batch_dict in train_loader:
                # if on GPU put mini-batch into CUDA memory
                batch_dict['corr_' + self.encoder.name] = corrupt(batch_dict[self.encoder.name], self.corruption)
                if self.use_cuda:
                    batch_dict[self.encoder.name] = batch_dict[self.encoder.name].cuda()
                    batch_dict['corr_'+self.encoder.name] = batch_dict['corr_'+self.encoder.name].cuda()
                try:
                    epoch_loss += svi.step(batch_dict, sample_size=batch_dict['y'].size(0)) / \
                                  (N_data * MINIBATCH_SIZE // SUBSAMPLE_RATIO)
                except Exception as err:
                    logger.debug(f"The batch that causes problems has length "
                                 f"{batch_dict['y'].size(0)}: {err}")
                if math.isnan(epoch_loss):
                    logger.debug(f"Incorrect hyperparameters")
                    return pred_train_old

                pred_tr = self.classify(batch_dict)
                pred_train.append(pred_tr.detach().cpu().numpy())

            pred_train = np.concatenate(pred_train, axis=0)
            acc_tr = round(accuracy_score(dt_tr['y'], pred_train > TH_PRED)*100, 3)

            if acc_tr > 99.5:
                perfect_tr.append(acc_tr)
            if len(perfect_tr) > 30:  # to reduce overfitting
                break
        logger.debug(f"Training finished in {asMinutes(time.time() - start)}")
        return pred_train

    def calc_predVI(self, batch_size=MINIBATCH_SIZE, **dta_te):
        test_loader = create_loader(batch_size=batch_size, use_cuda=self.use_cuda, **dta_te)
        preds, z_embs, x_reconst = [], [], []
        for batch_dict in test_loader:
            pred = self.classify(batch_dict)
            reconstructed_x, z_loc_emb = self.reconstruct_x(batch_dict)
            preds.append(pred.detach().cpu().numpy())
            x_reconst.append(reconstructed_x.detach().cpu().numpy())
            z_embs.append(z_loc_emb.detach().cpu().numpy())

        predVI = np.concatenate(preds, axis=0)
        self.z_loc_embedding = np.concatenate(z_embs, axis=0)
        self.x_reconst = np.concatenate(x_reconst, axis=0)
        if dta_te['y'].size > 0:
            acc_te = round(accuracy_score(dta_te['y'], predVI > TH_PRED) * 100, 3)
            logger.debug(f"Test accuracy: {acc_te}")
        return predVI


def create_loader(batch_size=MINIBATCH_SIZE, use_cuda=False, **dta):
    """
        Create a PyTorch DataLoader from dictionary-like input.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        use_cuda : bool
            Whether to enable pinned memory for CUDA transfer.
        dta : dict
            Feature arrays keyed by name. Sparse matrices are converted to dense.
            If 'y' is present, it is treated as integer labels.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding batches as dictionaries {feature_name: tensor}.
        """

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.features_dict = {
                k: torch.Tensor(v.toarray() if isinstance(v, ss.spmatrix) else v)
                for (k, v) in dta.items() if k not in ['y']
            }
            if 'y' in dta.keys() and not dta['y'].size==0:
                self.features_dict['y'] = torch.Tensor(np.array(dta['y'])).to(torch.int64)

        def __getitem__(self, index):
            return dict((f_name, f_values[index]) for f_name, f_values in self.features_dict.items())

        def __len__(self):
            return max([len(x) for x in self.features_dict.values()]+[0])

    return DataLoader(MyDataset(), batch_size=batch_size, num_workers=0, pin_memory=use_cuda, shuffle=False)