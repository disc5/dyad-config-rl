function [I2] = adjustBrightness(I, c)
%ADJUSTBRIGHTNESS Adjusts the brightness of a grayscale image
%   Adjusts the brightness by adding / subtracting intensity value c.
%   It is assumed, that the image is represented as gray scale values in
%   [0,1]. After the adjustment, the intenstity values are scaled between 0
%   and 1.
%   
%
%   Input:
%       I - grayscale images
%       c - parameter between -0.3 and 0.3
%
%   Output:
%       I2 - grayscale image with values between [0,1]
%
% (C) DS
    I2 = I+c;
    % truncate values > 1 to 1
    I2(I2>1) = 1;
    % truncate values < 1 to 0
    I2(I2<0) = 0;
end

