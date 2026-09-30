function [z,mu,sigma] = zscore(x,flag,dim)
if nargin < 2 || isempty(flag), flag = 0; end
if nargin < 3, dim = find(size(x) ~= 1, 1); if isempty(dim), dim = 1; end, end
mu = mean(x,dim); sigma = std(x,flag,dim); s0 = sigma; s0(s0==0) = 1;
z = (x - mu) ./ s0;
end
