function demo()
% it is a demo to train robust CNNs using mrelu with both the random
% training and mean training

saftype = 'mrelu'; % other choice: rbf1d
fprintf('******************************************************\nTraining MNIST CNNs...\n');
cnn_mnist_robust(false, saftype, true, true); % 1st parameter: true - only creating imdb
net = load(['data/mnist/', saftype, '-r-m/net-epoch-20.mat']); net = net.net;
imdb = load('data/mnist/imdb.mat'); images = imdb.images.data; labels = imdb.images.labels; data_mean = imdb.images.data_mean;

fprintf('******************************************************\nEvaluating MNIST CNNs...\n');
beta = [0, 0.01 : 0.01: 0.05, 0.1 : 0.05 : 0.3, 0.4, 0.5] ;
[pr, er] = cnn_eval_nsy(net, images, labels, beta); % noisy
[pr, er] = cnn_eval_adv(net, images, labels, beta); % adversarial
[pr, er] = cnn_eval_nss(net, data_mean, beta, 10000); % nonsense

fprintf('******************************************************\nTraining CIFAR CNNs ...\n');
cnn_cifar_robust(false, saftype, true, true); % 1st parameter: true - only creating imdb
net = load(['data/cifar/', saftype, '-r-m/net-epoch-90.mat']); net = net.net;
imdb = load('data/cifar/imdb.mat'); images = imdb.images.data; labels = imdb.images.labels; data_mean = imdb.images.data_mean;

fprintf('******************************************************\nEvaluating CIFAR CNNs...\n');
beta = [0, 0.01 : 0.005 : 0.05, 0.075, 0.1, 0.15];
[pr, er] = cnn_eval_nsy(net, images, labels, beta); % noisy
[pr, er] = cnn_eval_adv(net, images, labels, beta); % adversarial
[pr, er] = cnn_eval_nss(net, data_mean, beta, 10000); % nonsense

end
