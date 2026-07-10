imgsize = 200
batchsize = 1

params = {
    'batch_size': 1,
    'input_size': 200,  # 256
    'resize_scale': 286,
    'crop_size': 200,   # 256
    'fliplr': True,
    # model params
    'num_epochs': 2,
    'decay_epoch': 100,
    'ngf': 32,      # number of generator filters
    'ndf': 64,      # number of discriminator filters
    'num_resnet': 3,
    'lrG': 0.0001,
    'lrD': 0.0001,
    'beta1': 0.5,   # beta1 for Adam optimizer
    'beta2': 0.999, # beta2 for Adam optimizer
    'lambdaA': 10,  # lambdaA for cycle loss
    'lambdaB': 10,  # lambdaB for cycle loss
}