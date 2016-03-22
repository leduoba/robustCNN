% -------------------------------------------------------------------------
function adv_fg = cnn_fast_gradient(net, images, labels, nonsense)
% -------------------------------------------------------------------------
% nonsense: true - will choose the fastest gradient from 10 meaningful
% categories
% nonsense: false - will choose the fastest gradient from 9 meaningful
% categories rather than the true label
if nargin < 4, nonsense = false; end

for i = 1 : numel(net.layers), net.layers{i}.rememberOutput = true; end

adv_fg = zeros(size(images), 'single'); n = size(images, 4);

for i = 1 : 100 : n
    if ~mod(i + 99, 1000), fprintf('.'); end;
    indice = i : min(i + 99, n); adv_fg(:, :, :, indice) = fastgradient_once(net, images(:, :, :, indice), labels(1, indice), nonsense);
end
fprintf('\n');

% -------------------------------------------------------------------------
function gradient = fastgradient_once(net, im, labels, nonsense)
% -------------------------------------------------------------------------
% CNN_MNIST  Demonstrated MatConNet on MNIST
if nargin < 4, nonsense = false; end
if nonsense, targetN = 10; else targetN = 9; end

for i = 1 : numel(net.layers), net.layers{i}.rememberOutput = true; end
sz = size(im); gradient = zeros(sz(1), sz(2), sz(3), sz(4), 'single');

ims = zeros(size(im, 1), size(im, 2), size(im, 3), targetN * size(im, 4), 'single');
labls = zeros(1, targetN * size(im, 4), 'single');
for i = 1 : size(im, 4)
    ims(:, :, :, (i - 1) * targetN + 1 : i * targetN) = repmat(im(:, :, :, i), [1, 1, 1, targetN]);
    labls(1, (i - 1) * targetN + 1 : i * targetN) = setdiff(1 : 10, labels(i));
end

res = process_epoch(single(ims), single(labls), net);
for i = 1 : size(im, 4)
    nm = zeros(1, targetN);
    for j = 1 : targetN
        t = squeeze(res(1).dzdx(:, :, :, (i - 1) * targetN + j)); nm(j) = gather(sum(abs(t(:))));
    end
    [~, ind] = max(nm); gradient(:, :, :, i) = gather(res(1).dzdx(:, :, :, (i - 1) * targetN + ind));
end

% -------------------------------------------------------------------------
function  res = process_epoch(im, labels, net)
% -------------------------------------------------------------------------

% if using the GPU mode
% im = gpuArray(im) ; net = vl_simplenn_move(net, 'gpu') ;

% evaluate the CNN
net.layers{end}.class = labels ; dzdy = single(1) ;
res = vl_simplenn(net, im, dzdy, [], 'accumulate', false, ...
    'mode', 'normal', ...
    'conserveMemory', false, ...
    'backPropDepth', +inf, ...
    'sync', false, ...
    'cudnn', true) ;
