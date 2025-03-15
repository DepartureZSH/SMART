import math
import torch
import torch.multiprocessing
import torch.optim as optim
from torch.optim import Optimizer

class WarmupReduceLROnPlateau(object):
    def __init__(
            self,
            optimizer,
            gamma=0.5,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            T_max=100,
            eta_min=0,
            last_epoch=-1,
            patience=2,
            threshold=1e-4,
            cooldown=1,
            logger=None,
    ):
        if warmup_method not in ("constant", "linear", "cosine"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.logger = logger
        self.T_max = T_max  # Max number of iterations for cosine annealing
        self.eta_min = eta_min  # Minimum learning rate

        # The following code is copied from Pytorch=1.2.0
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        warmup_factor = 1
        # during warming up
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear" or self.warmup_method == "cosine":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                * self.gamma ** self.stage_count
                for base_lr in self.base_lrs
            ]
        elif self.warmup_method == "cosine":
            # Cosine Annealing: Adjust the learning rate using cosine function
            cos_anneal_lr = [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]
            return [
                lr 
                * warmup_factor 
                * self.gamma ** self.stage_count 
                for lr in cos_anneal_lr
            ]
        else:
            return [
                base_lr
                * warmup_factor
                * self.gamma ** self.stage_count
                for base_lr in self.base_lrs
            ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # The following part is modified from ReduceLROnPlateau
        if metrics is None:
            # not conduct validation yet
            pass
        else:
            # s = '=' * 40
            # print(f'{s} Try Decay {s}')
            if float(metrics) > (self.best + self.threshold):
                self.best = float(metrics)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs >= self.patience:
                # print(f'{s} Do Decay {s}')
                if self.logger is not None:
                    self.logger.info("Trigger Schedule Decay, RL has been reduced by factor {}".format(self.gamma))
                self.stage_count += 1  # this will automatically decay the learning rate
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Get Optimizer
def build_optimizer(model, opts):
    """ Re linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'classifier' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'classifier' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)
    return optimizer