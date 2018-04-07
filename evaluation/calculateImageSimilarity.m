function [similarity] = calculateImageSimilarity(img1, img2)
%EVALUATEIMAGESIMILARITY Wrapper for diverse image similarity measures
%   This function is used to enable better experimentation. It can be used
%   to exchange the similarity measure for determining, which op-chain
%   result is preferred over the other and for evaluation purposes.
    
    cfg = getConfig();
    
    if cfg.similarity_measure == cfg.SIMILARITY_SSIM
        similarity = ssim(img1, img2);
    elseif cfg.similarity_measure == cfg.SIMILARITY_FSIM
        similarity = FeatureSIM(img1, img2);
    elseif cfg.similarity_measure == cfg.SIMILARITY_MANHATTAN
        Z=abs(img1(:) - img2(:));
        similarity = 1 - sum(Z)/length(Z);
    else
        error('To be implemented.');
        similarity = -1;
    end
end

