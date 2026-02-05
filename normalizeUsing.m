function X_norm = normalizeUsing(X, mu, sigma)
% NORMALIZEUSING Normalize X using previously computed mu and sigma
X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);
end
