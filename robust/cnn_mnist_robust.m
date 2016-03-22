function cnn_mnist_robust(createImdbOnly, safType, randTraining, meanTraining)
% Demonstrates robust CNNs on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','mnist') ;
opts.dataDir = fullfile('data','mnist-data') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBatchNorm = true ;
opts.train = struct() ;

if nargin < 3, randTraining = true; meanTraining = true; createImdbOnly = false; end
if randTraining, modelname = [safType, '-r']; end
if meanTraining, modelname = [modelname, '-m']; end
modelpath = ['data/mnist/', modelname];

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
if createImdbOnly, return; end

net = cnn_mnist_init_robust(safType, meanTraining) ;

if randTraining
    trainset = find(imdb.images.set == 1); trainset1 = trainset(find(rand(1, numel(trainset)) > 0.5));
    imdb.images.data(:, :, 1, trainset1) = imdb.images.data(:, :, 1, trainset1) + ...
        single(255 * 0.50 * sign(randn(size(imdb.images.data(:, :, 1, trainset1)))));
end

if meanTraining
    imdb.meta.classes = [imdb.meta.classes, 'n', 'n'];

    sz = size(imdb.images.data); zeron = 6000;
    data1 = zeros(sz(1), sz(2), sz(3), sz(4) + zeron, 'single'); data1(:, :, :, 1 : sz(4)) = imdb.images.data;
    data1(:, :, :, sz(4) + 1 : end) = randn(sz(1), sz(2), sz(3), zeron, 'single');
    for i = 1 : zeron, data1(:, :, :, sz(4) + i) = data1(:, :, :, sz(4) + i) * 255 * rand; end
    imdb.images.data = data1; clear data1;
    imdb.images.set = [imdb.images.set, ones(1, zeron)];
    imdb.images.labels = [imdb.images.labels, zeros(1, zeron) + 11];

    sz = size(imdb.images.data); rubbn = 6000;
    data1 = zeros(sz(1), sz(2), sz(3), sz(4) + rubbn, 'single');
    data1(:, :, :, end - rubbn + 1 : end) = randn(sz(1), sz(2), sz(3), rubbn, 'single') * 127.5 + 127.5 ...
        - repmat(imdb.images.data_mean, [1, 1, 1, rubbn]);
    data1(:, :, :, 1 : sz(4)) = imdb.images.data;
    imdb.images.data = data1; clear data1;
    imdb.images.set = [imdb.images.set, ones(1, rubbn)];
    imdb.images.labels = [imdb.images.labels, zeros(1, rubbn) + 12];
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, getBatch, ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

mkdir(modelpath); copyfile([opts.expDir, '/net*'], modelpath);

% --------------------------------------------------------------------
function fn = getBatch()
% --------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y) ;

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
