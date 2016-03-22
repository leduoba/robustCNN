function [pr, er, images, adv_ns] = cnn_eval_nss(net, data_mean, beta, sampleN)
% evaluating robustness against nonsense samples

sz = size(data_mean); if size(sz) < 3, sz = [sz, 1]; end

images = randn(sz(1), sz(2), sz(3), sampleN, 'single');
for j = 1 : sampleN
    tim = images(:, :, :, j); images(:, :, :, j) = (tim - min(tim(:))) / (max(tim(:)) - min(tim(:))) * 255 - data_mean;
end
labels = zeros(1, sampleN) + 11;
adv_ns = cnn_fast_gradient(net, images, labels, true);

categoryN = 12; needexp = false; if strcmp(net.layers{end}.type, 'softmax'), needexp = true; end

% if using the GPU mode
% net = vl_simplenn_move(net, 'gpu') ;

sz = size(images); pr = zeros(sz(4), numel(beta), categoryN);
for i = 1 : sampleN
    if ~mod(i, 1000), fprintf('.'); end;
    ims_nss = zeros(sz(1), sz(2), sz(3), numel(beta), 'single');
    for j = 1 : numel(beta)
        ims_nss(:, :, :, j) = images(:, :, :, i) - sign(adv_ns(:, :, :, i)) * 255 * beta(j);
    end
    
% if using the GPU mode
% ims_nss = gpuArray(ims_nss);

    pre = process_epoch(ims_nss, repmat(10, [1, numel(beta)]), net);
    if needexp, pre = exp(pre); pre = pre ./ repmat(sum(pre, 2), [1, size(pre, 2)]); end;
    categoryN = size(pre, 2); pr(i, :, 1 : categoryN) = pre;
end
fprintf('\n');
pr = pr(:, :, 1 : categoryN); er = zeros(numel(beta), 2);
for i = 1 : numel(beta)
    a = squeeze(pr(:, i, :)); [maxv, ind] = max(a, [], 2);
    if categoryN > 10, er(i, 1) = mean(ind' < 11); else er(i, 1) = mean(maxv' > 0.5 ); end
    er(i, 2) = mean(maxv);
end;
disp([beta; er(:, 1)']);

% -------------------------------------------------------------------------
function  [pre, res] = process_epoch(im, labels, net)
% -------------------------------------------------------------------------

% if using the GPU mode
% im = gpuArray(im); net = vl_simplenn_move(net, 'gpu') ;

% evaluate the CNN
net.layers{end}.class = labels ;
res = vl_simplenn(net, im, [], [], 'accumulate', false, ...
    'mode', 'test', ...
    'conserveMemory', false, ...
    'backPropDepth', +inf, ...
    'sync', false, ...
    'cudnn', true) ;

% accumulate training errors
pre = squeeze(gather(res(end-1).x))';
