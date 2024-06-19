#python3 main.py --multirun +model=spectral model.ksize=[3,3],[5,5],[10,10],[15,15],[20,20],[25,25],[28,28] train.opt.lr=0.005 #the setup taken from original code
python3 main_interpol.py --multirun +model=cnn,spectral model.ksize=[10,10],[25,25] train.opt.lr=0.005 +im_shape=[8,8],[13,13],[18,18],[23,23],[28,28],[33,33],[38,38],[43,43],[48,48],[53,53],[58,58] #used for generating first plot 2024/06/18
#python3 main_interpol.py --multirun +model=spectral model.ksize=[10,10] train.opt.lr=0.005 +im_shape=[53,53],[58,58] #used for rest of above
#python3 main_interpol.py --multirun +model=spectral model.ksize=[25,25] train.opt.lr=0.005 +im_shape=[8,8],[13,13],[18,18],[23,23],[28,28],[33,33],[38,38],[43,43],[48,48],[53,53],[58,58] #used for rest of above

#python3 main_interpol.py --multirun +model=resnet +num_classes=10 train.opt.lr=0.005 +im_shape=[28,28]