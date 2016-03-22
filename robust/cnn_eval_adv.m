function [pr, er, adv_fg] = cnn_eval_adv(net, images, labels, beta)
% evaluating robustness against adversarial samples

% if using the GPU mode
% net = vl_simplenn_move(net, 'gpu') ;

adv_fg = cnn_fast_gradient(net, images, labels);

categoryN = 12; needexp = false; if strcmp(net.layers{end}.type, 'softmax'), needexp = true; end

sz = size(images); pr = zeros(sz(4), numel(beta), categoryN);
for i = 1 : size(images, 4)
    if ~mod(i, 1000), fprintf('.'); end;
    ims_adv = repmat(images(:, :, :, i), [1, 1, 1, numel(beta)]);
    for j = 1 : numel(beta)
        ims_adv(:, :, :, j) = ims_adv(:, :, :, j) - sign(adv_fg(:, :, :, i)) * 255 * beta(j);
    end

% if using the GPU mode    
% ims_adv = gpuArray(ims_adv);

    pre = process_epoch(ims_adv, repmat(labels(i), [1, numel(beta)]), net);
    if needexp, pre = exp(pre); pre = pre ./ repmat(sum(pre, 2), [1, size(pre, 2)]); end;
    categoryN = size(pre, 2); pr(i, :, 1 : categoryN) = pre;
end
fprintf('\n');
pr = pr(:, :, 1 : categoryN);
er = zeros(numel(beta), 2);
for i = 1 : numel(beta)
    a = squeeze(pr(:, i, :)); [maxv, ind] = max(a, [], 2);
    er(i, 1) = mean(ind' ~= labels); er(i, 2) = mean(maxv);
end;
disp([beta; er(:, 1)']);

% -------------------------------------------------------------------------
function  pre = process_epoch(im, labels, net)
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

pre = squeeze(gather(res(end-1).x))';
