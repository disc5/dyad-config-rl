function [total_error,restored,allIntermediates ] = evaluatePolicy(policy_model, distorted, originals)
%EVALUATEPOLICY Benchmarks a (trained) policy model
%   This function evaluates the performance of a given policy model.
%
%   Notes:
%   - The pipeline length is specified in getConfig().
%   - The performance is determined by the similarity of the ground truth
%     image and the produced output of the pipeline. 
%
    N = size(distorted,1);
    restored = cell(N,1);
    allIntermediates = cell(N,1);
    
    total_error = 0;
    for i3 = 1 : N
        distortedImage = distorted{i3};
        originalImage = originals{i3};
        [processedImage, op_seq, op_seq_ids, intermediates] = applyPolicy(policy_model, distortedImage);
        restored{i3} = processedImage;
        allIntermediates{i3} = intermediates;
        
        sim_quality = calculateImageSimilarity(processedImage,originalImage);
        if mod(i3,200)==1
            fprintf('Img %d: sim = %3.4f,\t', i3, sim_quality);
            for i5 = 1 : length(op_seq_ids)
                fprintf('Op_%d param=%3.4f,\t', op_seq_ids{i5}, op_seq{i5});
            end
            fprintf('\n');
        end
        if isnan(sim_quality)
            sim_quality = 0;
        end
        total_error = total_error + (1-sim_quality);
    end
    total_error = total_error / N;
end

