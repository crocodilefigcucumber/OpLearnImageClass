#python3 main_interpol.py --multirun +model=cnn,spectral model.ksize=[10,10],[25,25] train.opt.lr=0.005 +im_shape=[8,8],[13,13],[18,18],[23,23],[28,28],[33,33],[38,38],[43,43],[48,48],[53,53],[58,58] #used for generating first plot 2024/06/18
#python3 main_interpol.py --multirun +model=spectral model.ksize=[25,25] train.opt.lr=0.005 +im_shape=[28,28]
#python3 main_interpol.py --multirun +model=resnet train.opt.lr=0.005 +im_shape=[28,28]
#python3 main_interpol.py --multirun +model=cnn model.ksize=[10,10],[25,25] train.opt.lr=0.002 +im_shape=[13,13],[18,18],[23,23],[28,28],[33,33],[38,38],[43,43],[48,48],[53,53],[58,58]
#python3 main_interpol.py --multirun +model=spectral model.ksize=[10,10],[25,25] train.opt.lr=0.002 +im_shape=[13,13],[18,18],[23,23],[28,28],[33,33],[38,38],[43,43],[48,48],[53,53],[58,58]
python3 main_interpol.py --multirun +model=fno train.opt.lr=0.002 +im_shape=[28,28] model.N_layers=3,5,7 model.hidden_channels=3,5,7 model.n_modes=3,5,7
