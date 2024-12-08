# Experiments on FMNIST for ETHZ Semester Paper
[![View PDF](https://img.shields.io/badge/View-PDF-red)](./experiments/interpolations/plots/ModelsVsCNNsOnNativeResolutions,BILINEAR.pdf)


Original Setup has been taken from [here](https://github.com/samirak98/FourierImaging), [arxiv](https://arxiv.org/abs/2304.01227), and slight modifications (incl. FNO/CNO implementation) of the codebase have been performed.
For CNO implementation refer to [this repo](https://github.com/camlab-ethz/ConvolutionalNeuralOperator), ViTs have been fine-tuned using code from [this repo](https://github.com/bwconrad/vit-finetune).

To reproduce experiments for everything other than ViT, simply run the (commented-out) parts of the run script.
To reproduce ViT finetuning, clone [this repo](https://github.com/bwconrad/vit-finetune), first run `preprocess_FMNIST.py`, then the run script, and for evalutation, run `vit_eval_script.py`.
