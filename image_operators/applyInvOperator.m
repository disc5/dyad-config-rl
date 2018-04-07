function [I2] = applyInvOperator(op_id, param_value, I)
%APPLYOPERATOR Summary of this function goes here
%   Detailed explanation goes here
        if op_id == 1 % Log
            [I2] = invLogarithmicOperator(I, param_value);
        elseif op_id == 2  % Gamma
            [I2] = invGammaOperator(I, param_value);
        elseif op_id == 3 % Bright
            [I2] = invAdjustBrightness(I, param_value);
        end
end

