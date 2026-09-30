function y = prctile(x,p)
x = sort(x(~isnan(x(:)))); n = numel(x);
if n == 0, y = nan(size(p)); return, end
q = 100*((1:n)-0.5)/n;
y = zeros(size(p));
for k = 1:numel(p)
  if p(k) <= q(1), y(k) = x(1);
  elseif p(k) >= q(end), y(k) = x(end);
  else, y(k) = interp1(q, x, p(k)); end
end
end
