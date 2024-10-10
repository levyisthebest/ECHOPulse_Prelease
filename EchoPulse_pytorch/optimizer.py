from torch.optim import AdamW, Adam, lr_scheduler


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


class LinearWarmup_CosineAnnealing():
    """ Construct two LR schedulers.
    First one is for a linear warmup.
    Second one is for a cosine annealing."""

    def __init__(self, optimizer,  # optimizer to schedule
                 linear_warmup_start_factor, linear_warmup_total_iters,  # linear warmup
                 cosine_annealing_T_max, cosine_annealing_eta_min  # cosine annealing
                 ):

        self.scheduler_linear = lr_scheduler.LinearLR(optimizer,
                                                      # The number we multiply learning rate in the first epoch
                                                      start_factor=linear_warmup_start_factor,
                                                      total_iters=linear_warmup_total_iters)  # The number of iterations that multiplicative factor reaches to 1

        self.scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer,
                                                               # Maximum number of iterations.
                                                               T_max=cosine_annealing_T_max,
                                                               eta_min=cosine_annealing_eta_min)  # Minimum learning rate.

        self.switch = linear_warmup_total_iters

    def step(self, nb_steps):

        if nb_steps <= self.switch:
            self.scheduler_linear.step()
        else:
            self.scheduler_cosine.step()

        return
