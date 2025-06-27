from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer

def getOptimizer(args, model):

    """
    Function to select optimizer

    INPUTS:-
    1) args: The training/testing arguments
    2) model: The trainig model

    OUTPUTS:-
    1) optim: The optimzer
    """

    argsOptim = SimpleNamespace()

    if(args.optimizer == 'adam'):
        argsOptim.weight_decay = 0
        argsOptim.lr = args.lr
        argsOptim.opt = 'adam'
        argsOptim.momentum = 0.9
        args.eps = 1e-7

        optim = create_optimizer(argsOptim, model)
        return optim
    
    if(args.optimizer == 'nadam'):
        argsOptim.weight_decay = 0
        argsOptim.lr = args.lr
        argsOptim.opt = 'adam'
        argsOptim.momentum = 0.9
        args.eps = 1e-7

        optim = create_optimizer(argsOptim, model)
        return optim

    #if(args.optimizer == 'adao')
    
    
