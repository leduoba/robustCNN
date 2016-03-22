function cnn_cifar_robust(createImdbOnly, safType, randTraining, meanTraining)
% Demonstrates robust CNNs on CIFAR-10

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data', 'cifar') ;
opts.dataDir = fullfile('data','cifar-data') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = false ; % true ;
opts.contrastNormalization = false ; % true ;
opts.train = struct() ;

if nargin < 3, randTraining = true; meanTraining = true; createImdbOnly = false; end
if randTraining, modelname = [safType, '-r']; end
if meanTraining, modelname = [modelname, '-m']; end
modelpath = ['data/cifar/', modelname];

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
if createImdbOnly, return; end

net = cnn_cifar_init_robust(safType, meanTraining) ;

if meanTraining
    imdb.meta.classes{end + 1} = 'nonsense'; imdb.meta.classes{end + 1} = 'nonsense';

    sz = size(imdb.images.data); zeron = 2500;
    data1 = zeros(sz(1), sz(2), sz(3), sz(4) + zeron, 'single'); data1(:, :, :, 1 : sz(4)) = imdb.images.data;
    data1(:, :, :, sz(4) + 1 : end) = randn(sz(1), sz(2), sz(3), zeron, 'single');
    for i = 1 : zeron, data1(:, :, :, sz(4) + i) = data1(:, :, :, sz(4) + i) * 64 * rand; end
    imdb.images.data = data1; clear data1;
    imdb.images.set = [imdb.images.set, ones(1, zeron)];
    imdb.images.labels = [imdb.images.labels, zeros(1, zeron) + 11];

    sz = size(imdb.images.data); rubbn = 2500;
    data1 = zeros(sz(1), sz(2), sz(3), sz(4) + rubbn, 'single');
    data1(:, :, :, end - rubbn + 1 : end) = randn(sz(1), sz(2), sz(3), rubbn, 'single') * 96 + 127.5 ...
        - repmat(imdb.images.data_mean, [1, 1, 1, rubbn]);
    data1(:, :, :, 1 : sz(4)) = imdb.images.data;
    imdb.images.data = data1; clear data1;
    imdb.images.set = [imdb.images.set, ones(1, rubbn)];
    imdb.images.labels = [imdb.images.labels, zeros(1, rubbn) + 12];
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, getBatch(randTraining), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

mkdir(modelpath); copyfile([opts.expDir, '/net*'], modelpath);


% -------------------------------------------------------------------------
function fn = getBatch(randTraining)
% -------------------------------------------------------------------------
if randTraining, fn = @(x,y) getRobustCNNBatch_randTraining(x,y) ; else fn = @(x,y) getRobustCNNBatch(x,y); end

% -------------------------------------------------------------------------
function [images, labels] = getRobustCNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function [images, labels] = getRobustCNNBatch_randTraining(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if unique(imdb.images.set(batch)) == 1 %  add random noise only in training
	if rand > 0.5, images = images + 255 * 0.15 * sign(randn(size(images), 'single')); end
end

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ; 
  z = bsxfun(@minus, z, mean(z,1)) ; 
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ; 
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ; 
  W = z(:,set == 1)*z(:,set == 1)'/60000 ; 
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ; 
  en = sqrt(mean(d2)) ; 
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;

imdb.images.data_mean = dataMean;

imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
