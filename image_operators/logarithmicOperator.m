function [I2] = logarithmicOperator(I,c)
%LOGARITHMICOPERATOR Logarithmic operator for grayscale images
%   "The dynamic range of an image can be compressed by replacing each pixel 
%   value with its logarithm. This has the effect that low intensity pixel 
%   values are enhanced. Applying a pixel logarithm operator to an image 
%   can be useful in applications where the dynamic range may too large to be 
%   displayed on a screen (or to be recorded on a film in the first place)." 
%   Quote by https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
%
%   Input:
%       I - grayscale image
%       c - parameter between 1 and 1.5
%
%   Output:
%       I2 - transformed image
%
%
% (C) 2018 Dirk Schaefer
    %imax = max(I(:));
    %c = 1.0/log(1+imax);
    I2 = c.*log(1+abs(I));
end

