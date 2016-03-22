# robustCNN
robust CNNs using symmetric activation functions

1. Make sure you have installed and set up MatConvnet correctly;
3. Copy the folder 'robust' into '[MatConvnet_root]/examples';
2. Enter the folder '[MatConvnet_root]/examples/robust' and run 'robust_setup.m';
3. Run 'demo.m' for both training and evaluating.

Moreover, there are pre-trained models in the folder 'pre-trained'. You can evaluate them without training by
(1). Setting 'createImdbOnly = true' (the 1st parameter) for cnn_mnist_robust and cnn_cifar_robust in line 7 and 21;
(2). Copy all folders in 'pre-trained' into 'data';
(3). Run 'demo.m' for only evaluating.

Related Paper:

Suppressing the unusual: towards robust CNNs using symmetric activation functions. You can donwnload it at http://arxiv.org/abs/1603.05145.

Please feel free to contact me: zhaoqy@buaa.edu.cn
