function [I2] = gammaOperator(I,c)
%gammaOperator Gamma operator for grayscale images
%
%   Input:
%       I - grayscale image
%       c - parameter between -1 and 1
%
%   Output:
%       I2 - transformed image
%
%
% (C) 2018 Dirk Schaefer

    I2 = I.^(c+1);
end

