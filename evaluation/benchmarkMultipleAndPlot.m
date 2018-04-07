function [results] = benchmarkMultipleAndPlot(policy_model, distorted, originals, selected_ids)
%BENACHMARKPLOTMULTIPLE Benchmarks policy model on multiple images
%   Detailed explanation goes here

    N = length(selected_ids);
    results = cell(N,1);
    for i1 = 1 : N
        id = selected_ids(i1);
        [result] = benchmarkModel(policy_model, distorted{id}, originals{id});
        results{i1} = result;
    end

    % Subplots: Nx4
    figure
    cnt = 1;
    for i1 = 1 : N
        result = results{i1};
        
        subplot(N,4,cnt)
        imshow(result.images{1});
        %title('Input');
        %xlabel(num2str(result.similarities{1},3))

        cnt = cnt + 1;
        
        subplot(N,4,cnt)
        imshow(result.images{2})
        %title(['Op1 (', num2str(result.params{1}),')']);
        title([ num2str(result.params{1})]);
        %xlabel(num2str(result.similarities{2},3))

        cnt = cnt + 1;
        
        subplot(N,4,cnt)
        imshow(result.images{3})
        %title(['Op2 (', num2str(result.params{2}),')']);
        title([num2str(result.params{2})]);
        %xlabel(num2str(result.similarities{3},3))

        cnt = cnt + 1;
        
        subplot(N,4,cnt)
        imshow(result.images{4})
        %title(['GT'])
        
        cnt = cnt + 1;
    end
    
end

