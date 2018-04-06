function [I2] = invLogarithmicOperator(I, c)
%INVLOGARITHMICOPERATOR Inverse Logarithmic Operator for grayscale images
%   This produces the inverse of the logarithmic operator, s.t.
%   an image I should be recovered approximately when 
%   it is applied on invLogOp(LogOp(I,c),c)
%
%   Input:
%       I - grayscale image with values between 0 and 1
%       c - parameter between 0.5:5 (in 0.5 steps)
%
%   Ouput:
%       I2 - transformed image
%
%   (C) 2018 Dirk Schaefer

    I2 = exp(I./c) - 1;
end

