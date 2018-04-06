function [Op_log_values, Op_gamma_values] = getOperatorParameterSpace()
%GETOPERATORPARAMETERSPACE Returns the parameters available for the different image
%ops
%   Detailed explanation goes here
    %Op_log_values = [1.5, 2, 2.5, 3, 3.5, 4]; % 0.5, 1
    Op_log_values = [1.5, 2, 2.5, 3];
    %Op_gamma_values = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0];
    Op_gamma_values = [1.0,1.2,1.4,1.6];

end

