#######################Data############################

ROUNDING_DIGITS = 4  # for precision_recall_data

#######################Train BNN Operational############################
LR_ADAM = 5e-4  # ADAM: learning rates
BETAS = (0.97, 0.999)  # for very stochastic models it may make sense to use higher values like (.95, .999)

HIDDEN_DIM = 80  # number of hidden variables of the encoder
Z_DIM = 80  # number of variables output by the encoder
MINIBATCH_SIZE = 32
SUBSAMPLE_RATIO = 2

N_EPOCHS_EVAL = 1000
N_EPOCHS_TRAIN = 100