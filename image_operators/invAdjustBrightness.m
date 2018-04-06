function [I2] = invAdjustBrightness(I, c)
%invAdjustBrightness Inverse Brightness Operator for grayscale images
%   This produces the inverse of the brightness operator, s.t.
%   an image I should be recovered approximately when 
%   it is applied on invBrightOp(BrigthOp(I,c),c)
%
%   Input:
%       I - grayscale images
%       c - parameter between -0.5 and 0.5
%
%   Output:
%       I2 - grayscale image with values between [0,1]
%
%   (C) 2018 Dirk Schaefer

    I2 = I-c;
    % truncate values > 1 to 1
    I2(I2>1) = 1;
    % truncate values < 1 to 0
    I2(I2<0) = 0;
end
