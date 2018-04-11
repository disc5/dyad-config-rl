function [Op_log_values, Op_gamma_values, Op_brightness_values] = getOperatorParameterSpace()
%GETOPERATORPARAMETERSPACE Returns the parameters available for the different image
%ops
%   Detailed explanation goes here
    Op_log_values = [1.5, 2, 2.5, 3];
    Op_gamma_values = [1.0,1.2,1.4,1.6];
    Op_brightness_values = [-0.2,-0.1,0.1,0.2];

end

