classdef plnet_debug<handle
    %PLNET Plackett-Luce Network for Preference Learning Problems
    %   PLNet is based on the Plackett-Luce model and can be used to 
    %   learn object orderings. This version of PLNet is based on a
    %   three-layered MLP (or single-hidden layer feedforward neural 
    %   network SLFN).
    %
    %   By Dirk Schaefer
    %
    %   Changelog:
    %   2017-01-20 - Added "Early Stopping Mechanism" (Prechelt) 
    %   2016-05-10 - Bugfix at determining the number of alternatives at getUtilityScoresOfDataset
    %   2016-05-05 - Fixed problem with incomplete ordering observations
    %   2016-05-01 - INIT
    
    properties
        numLayers
        layerSizes
        weights
        biases  
    end
    
    methods
        function obj = plnet_debug(layerSizes, initialWeightRange)
            if nargin == 0
                error('Layers must be specified, e.g. [6,10,1]')
            else
                obj.numLayers = length(layerSizes);
                obj.layerSizes = layerSizes;
                obj.weights = cell(obj.numLayers-1,max(layerSizes));
                obj.biases = cell(obj.numLayers-1,1);
                % Init weights and biases
                %rng(1)
                ub = initialWeightRange;
                lb = -ub;
                for i1 = 1 : obj.numLayers-1
                    for i2 = 1 : layerSizes(i1+1)
                        obj.weights{i1,i2} = lb+(ub-lb).*rand(1,layerSizes(i1));
                    end
                    obj.biases{i1} = lb+(ub-lb).*rand(1,layerSizes(i1+1));
                    if (i1==obj.numLayers-1)
                        obj.biases{i1} = 0;
                    end
                end
            end
        end
        
        function SGD(obj, dataset, nEpochs, eta)
            % Stochastic Gradient Descent Training
            % Inputs:
            %   dataset - cell array of N observations, i.e.
            %               Mn x d ordered vectors.
            %   epochs - number of training steps
            N = size(dataset,1);
            
            useValSet = 1;
            useEarlyStopping = 1; % requires useValSet to be activated
            earlyStoppingCheckFreq = 100; % check every #times if validation error has risen
            showPlot = 0;
            
            if (useValSet == 1)
                % Split data set into tr and va sets
                % Note: only meaningful in case of full rankings
                nVa = floor(N*0.2); % proportion 0.1, 0.5 (Prechelt)
                nTr = N-nVa;
                trSet = dataset(1:nTr,:);
                vaSet = dataset(nTr+1:end,:);
                histTr = nan(1,nEpochs);
                histVa = nan(1,nEpochs);
                for i1 = 1 : nEpochs
                    RP = randperm(nTr);
                    for i2 = 1 : nTr
                        ctObs = trSet{RP(i2)};
                        obj.processSingleObservation(ctObs, eta);
                    end
                    fprintf('Epoch %d: NLL(tr) :%3.4f / NLL(va) :%3.4f \n', i1, obj.getNLLDataset(trSet), obj.getNLLDataset(vaSet));
                    histTr(i1) = obj.getNLLDataset(trSet);
                    histVa(i1) = obj.getNLLDataset(vaSet);
                    
                    if (showPlot == 1)
                        title('PLNet Training')
                        plot(histTr,'-*','Color',[0.4941    0.8314    0.4078],'Linewidth',2)
                        hold on
                        plot(histVa,'--v','Color',[0.9647    0.2118    0.0471],'Linewidth',2)
                        legend('Tr','Internal Va','Location','northeast')
                        xlabel('Steps')
                        ylabel('NLL')
                        grid on
                        drawnow
                    end
                    % Early Stopping Mechanism (Prechelt)
                    if (useEarlyStopping == 1)
                        if (mod(i1, earlyStoppingCheckFreq) == 0)
                            if ( histVa(i1) >  histVa(i1-earlyStoppingCheckFreq + 1))
                                fprintf('Early Stop! Early stopping criteria applies at epoch %d. (Check Freq: %d)\n',i1, earlyStoppingCheckFreq);
                                break;
                            end
                        end
                    end
                end
            else
                 for i1 = 1 : nEpochs
                    RP = randperm(N);
                    for i2 = 1 : N
                        ctObs = dataset{RP(i2)};
                        obj.processSingleObservation(ctObs, eta);
                    end
                    fprintf('Epoch %d: NLL(tr) :%3.4f \n', i1, obj.getNLLDataset(dataset));
                end
            end
        end
        
        function processSingleObservation(obj, observation, eta)
            % Takes a single observation, i.e. an ordered list of vectors, 
            % collects activations by performing multiple forward passes (which
            % would ideally by performed in parallel).
            % It uses the last layer activations to calculate the item deltas.
            % Those are used then in backpropagation for updating weights.
            % 
            % Note: In this implementation the list of vectors must be in the
            % correct sequence, i.e. no additional support data structure like 
            % permutations are considered here.
            %
            % observation: Mn x p row vectors.
            Mn = size(observation,1);
            allacts = cell(Mn,2);
            lastActivations = zeros(Mn,1);
            for i1 = 1 : Mn
                ctDyad = observation(i1,:);
                [activations, zs] = obj.feedforward(ctDyad);
                
                allacts{i1,1} = activations;
                allacts{i1,2} = zs;
                lastActivations(i1) = activations{end};
            end
            deltas = zeros(Mn,1);
            
            % Update strategy 1: "chained"
%             for i1 = 1 : Mn
%                 deltas(i1) = obj.rankingCostDerivative(i1, lastActivations);
%                 activations = allacts{i1,1};
%                 zs = allacts{i1,2};
%                 [nabla_w, nabla_b] = obj.backpropagate(activations,zs,deltas(i1));
%                 obj.update(nabla_w, nabla_b, eta);
%             end
            
            % Update strategy 2: "parallel"
            % Note: this variant seems to me more plausable since the
            % "lastActivations" are calculated based on the master's NN
            % weights. And backpropagation on those weights makes sense.
            for i1 = 1 : Mn
                deltas(i1) = obj.rankingCostDerivative(i1, lastActivations);
            end
            
            nabla_ws = cell(Mn,1);
            nabla_bs = cell(Mn,1);
            for i1 = 1 : Mn
                activations = allacts{i1,1};
                zs = allacts{i1,2};
                [nabla_w, nabla_b] = obj.backpropagate(activations,zs,deltas(i1));
               
                %for i5 = 1 : 10
                %    nabla_w{1}(i5,:) = nabla_w{1}(i5,:) / norm([nabla_w{1}(i5,:),nabla_b{1}(i5)]);
                %end
                %nabla_w{2} = nabla_w{2}/norm(nabla_w{2});
                % -> wrong location for normalization !!
                nabla_ws{i1} = nabla_w;
                nabla_bs{i1} = nabla_b;
            end
       
            for i1 = 1 : Mn
                nabla_w = nabla_ws{i1};
                nabla_b = nabla_bs{i1};
                obj.update(nabla_w, nabla_b, eta);
            end
            
%             for i1 = 2 : 2
%                 nabla_w = nabla_ws{i1};
%                 nabla_b = nabla_bs{i1};
%                 obj.update(nabla_w, nabla_b, eta);
%             end
            
        end
       
        function update(obj, nabla_w, nabla_b, eta)
            % Update the net weights.
            % Note: the update does not affect the bias term of
            % the penultimate layer.
            for i1 = 1 : obj.numLayers-1
                for i2 = 1 : obj.layerSizes(i1+1)
                    %obj.weights{i1,i2} = obj.weights{i1,i2} - eta.*nabla_w{i1}(i2,:)/norm(nabla_w{i1}(i2,:));
                    obj.weights{i1,i2} = obj.weights{i1,i2} - eta.*nabla_w{i1}(i2,:);
                end
                if (i1<obj.numLayers-1)
                    %obj.biases{i1} = obj.biases{i1} - eta.*nabla_b{i1}/norm(nabla_b{i1});
                    obj.biases{i1} = obj.biases{i1} - eta.*nabla_b{i1};
                end
            end
        end
        
        function [activations, zs, utility] = feedforward(obj, input)
            % Performes the "feed forward" procedure on a network. 
            % input: a row vector
            %
            % TODO: improve this, because as it is now it set fixed to the following
            % architecture: input,1 hidden and output layer
            activations = cell(3,1);
            zs = cell(2,1);
            
            a = input';            
            activations{1} = a;
            
            % FORWARD first hidden layer
            A = zeros(obj.layerSizes(2),1);
            ZS = zeros(obj.layerSizes(2),1);
            for i2 = 1 : obj.layerSizes(2)
                ZS(i2) = (obj.weights{1,i2}*a)+obj.biases{1}(i2);
                A(i2) = obj.sigmoid(ZS(i2));
            end
            activations{2} = A;
            zs{1} = ZS;
            
            % FORWARD to output layer
            a = obj.weights{2,1}*A + obj.biases{2};
            zs{2} = a;
            activations{3} = a;    
            utility = a;
        end
        
        function [nabla_w, nabla_b] = backpropagate(obj, activations, zs, delta)
            % Backpropagates errors for single item of an observation (and
            % correspondingly a SINGLE network).
            %
            % TODO: this implementation assumes exactly ONE hidden layer
            nabla_w = cell(2,1);
            nabla_b = cell(2,1);
            nabla_b{2} = 0;%delta;
            nabla_w{2} = delta.*activations{2}';
            
            % calculate nabla_w and b for higher layers
            z = zs{1};
            sp = obj.sigmoid_prime(z);
            delta2 = (obj.weights{2}'.*delta).*sp;
            nabla_b{1} = delta2';
            nabla_w{1} = delta2*activations{1}';
        end
        
        function [utilities] = getUtilities(obj, observation)
            % Returns the utilities 
            % Inputs: 
            %   observation - a matrix Mn x p of row vectors
            %
            % Output:
            %   utilities - a row vector of Mn - many reals
            Mn = size(observation,1);
            utilities = zeros(1,Mn);
            for i1 = 1 : Mn
                [~,~,utilities(i1)] = obj.feedforward(observation(i1,:));
            end   
        end
        
        function [utilityMatrix] = getUtilityScoresOfDataset(obj, dataset)
            % Returns the utilities for a set of observations.
            % Input:
            %   dataset - a cell array of N observations
            %
            % Output:
            %   utilityMatrix - N x M utility values
            
            N = size(dataset,1);
            % M = max(cellfun('length',dataset)) % wrong: this gives the
            % columns (=dimensions)
            % determine max number of alternatives used in the data set
            M = -1;
            for i1 = 1 : N
                [ctM,~] = size(dataset{1});
                if (ctM>M)
                    M = ctM;
                end
            end
            utilityMatrix = ones(N,M).*-1;
            for i1 = 1 : N
                ctUtilities = obj.getUtilities(dataset{i1});
                utilityMatrix(i1,1:length(ctUtilities)) = ctUtilities;
            end
        end
        
        function [nllData] = getNLLDataset(obj, dataset)
            % Returns the NLL of a set of observations.
            
            UMAT = obj.getUtilityScoresOfDataset(dataset);
            N = size(UMAT,1);
            nllData = 0;
            for i1 = 1 : N
                Mn = length(UMAT(i1,UMAT(i1,:)~=-1));
                nllData = nllData + obj.getNLLValue(UMAT(i1,1:Mn));
            end
            nllData = nllData / N;
        end
        
        
        function [cost] = rankingCostDerivative(~, rankPosition, observationOutputActiviations)
            % Returns the cost derivative for an item at a certain rank
            % position for learning Plackett-Luce MLE. 
            % The variable "observationOutputActiviations" refers to the output
            % activation of each single observation item (those would ideally be acquired in
            % parallel).
            %
            % Note: this implementation assumes that that the output_activations
            % of an observation are in the "correct" or desired ordering already. 
            cost = 0;
            M = length(observationOutputActiviations);
            for k = 1 : M-1
                if (rankPosition >= k)
                    denom = 0;
                    for l = k : M
                        denom = denom + exp(observationOutputActiviations(l));
                    end
                    cost = cost + exp(observationOutputActiviations(rankPosition))/denom;
                end
            end
            if (rankPosition < M)
                cost = cost - 1;
            end
        end
        
        function [nll] = getNLLValue(~, utilities)
            % Plackett-Luce NLL
            nll = 0;
            M = length(utilities);
            for i1 = 1 : M-1
                logsum = 0;
                for i2 = i1 : M
                   logsum = logsum + exp(utilities(i2));
                end
                nll = nll + log(logsum);
            end
            
            for i1 = 1 : M-1
                nll = nll - utilities(i1);
            end
        end
        
        function [val] = sigmoid(~, z)
            % Computes the sigmoid function.
            val =  1./(1+exp(-z));
        end
        
        function [val] = sigmoid_prime(obj, z)
            % Computes the derivative of the sigmoid function.
            val = obj.sigmoid(z).*(1-obj.sigmoid(z));
        end
        
        function new = copy(this)
            % Instantiate new object of the same class.
            new = plnet(this.layerSizes);

            % Copy all non-hidden properties.
            p = properties(this);
            for i = 1:length(p)
                new.(p{i}) = this.(p{i});
            end
        end
        
    end
    
end

