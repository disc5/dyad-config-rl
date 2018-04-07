function [I2] = applyOperator(op_id, param_value, I)
%APPLYOPERATOR Summary of this function goes here
%   Detailed explanation goes here
        if op_id == 1 % Log
            [I2] = logarithmicOperator(I, param_value);
        elseif op_id == 2 % Gamma
            [I2] = gammaOperator(I, param_value);
        elseif op_id == 3 % Bright 
            [I2] = adjustBrightness(I, param_value);
        end
end

