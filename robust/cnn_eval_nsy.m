function [pr, er] = cnn_eval_nsy(net, images, labels, beta)
% evaluating robustness under gaussian noise

% if using the GPU mode
% net = vl_simplenn_move(net, 'gpu') ;

categoryN = 12; needexp = false; if strcmp(net.layers{end}.type, 'softmax'), needexp = true; end
sz = size(images); pr = zeros(sz(4), numel(beta), categoryN);

for i = 1 : size(images, 4)
    if ~mod(i, 1000), fprintf('.'); end;
    noise = sign(randn(sz(1 : 3)));
    ims_nsy = repmat(images(:, :, :, i), [1, 1, 1, numel(beta)]);

    for j = 1 : numel(beta)
        ims_nsy(:, :, :, j) = ims_nsy(:, :, :, j) - noise * 255 * beta(j);
    end

% if using the GPU mode    
% ims_adv = gpuArray(ims_nsy);
    
    pre = process_epoch(ims_nsy, repmat(labels(i), [1, numel(beta)]), net);
    if needexp, pre = exp(pre); pre = pre ./ repmat(sum(pre, 2), [1, size(pre, 2)]); end;
    categoryN = size(pre, 2); pr(i, :, 1 : categoryN) = pre;
end
fprintf('\n');
pr = pr(:, :, 1 : categoryN); er = zeros(numel(beta), 2);
for i = 1 : numel(beta)
    a = squeeze(pr(:, i, :)); [maxv, ind] = max(a, [], 2);
    er(i, 1) = mean(ind' ~= labels); er(i, 2) = mean(maxv);
end;
disp([beta; er(:, 1)']);

% -------------------------------------------------------------------------
function  [pre, res] = process_epoch(im, labels, net)
% -------------------------------------------------------------------------

% if using the GPU mode
% im = gpuArray(im) ; net = vl_simplenn_move(net, 'gpu') ;

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
