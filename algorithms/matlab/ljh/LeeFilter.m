function LeeImages = LeeFilter(Im, n, ENL, functionMode)
% LeeFilter applies a classic Lee speckle filter.
%
% Input:
%   Im           input image
%   n            half window size, use 3 or 4
%   ENL          equivalent number of looks
%   functionMode 'r' for ratio model, 'd' for difference model
%
% Output:
%   LeeImages    filtered image

    if nargin < 4 || isempty(functionMode)
        functionMode = 'r';
    end
    if nargin < 3 || isempty(ENL)
        ENL = 4.5;
    end
    if nargin < 2 || isempty(n)
        n = 4;
    end

    Im = double(Im);
    winSize = 2 * n + 1;
    kernel = ones(winSize, winSize) / (winSize * winSize);

    localMean = imfilter(Im, kernel, 'symmetric');
    localMeanSq = imfilter(Im .^ 2, kernel, 'symmetric');
    localVar = max(localMeanSq - localMean .^ 2, 0);

    switch lower(functionMode)
        case 'd'
            noiseVar = mean(localVar(:)) / max(ENL, eps);
            k = max(localVar - noiseVar, 0) ./ max(localVar, eps);
        otherwise
            sigma_v2 = 1 / max(ENL, eps);
            signalVar = max((localVar - (localMean .^ 2) * sigma_v2) ./ (1 + sigma_v2), 0);
            k = signalVar ./ max(localVar, eps);
    end

    LeeImages = localMean + k .* (Im - localMean);
end
