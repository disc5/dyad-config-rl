function [I2, stop] = applyOperator(op_id, param_value, I)
%APPLYOPERATOR Summary of this function goes here
%   Detailed explanation goes here
        stop = false;
        if op_id == 1 % Log
            [I2] = logarithmicOperator(I, param_value);
        elseif op_id == 2 % Gamma
            [I2] = gammaOperator(I, param_value);
        elseif op_id == 3 % Bright 
            [I2] = adjustBrightness(I, param_value);
        elseif op_id == 4 % Unsharping mask filter - has the effect of making edges and fine details more crisp
            h = fspecial('unsharp');
            I2 = imfilter(I,h);
            % truncate values > 1 to 1
            I2(I2>1) = 1;
            % truncate values < 1 to 0
            I2(I2<0) = 0;
        elseif op_id == 5 % Hisogram Equalization
            I2 = histeq(I);
        elseif op_id == 6 % Stop Operator
            stop = true;
            I2 = I;
        end
end

