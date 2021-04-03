% This is a function that given a set of validation data and a binary
% decision tree will select the non-leaf node of the tree to remove such 
% that validation accuracy increases by the greatest amount.  (Note that 
% sometimes removing any non-leaf node can only hurt things.  In this case,
% this function selects the non-leaf node that will result in the smallest
% decrease in validation accuracy). 
%
% Usage: dTOut = pruneSingleGreedyNode(x, y, dT)
%
% Inputs: 
%
%   x - a S by D matrix, where S is the number of samples and D is the
%   length of a feature vector.   x(s,:) gives the feature vector for 
%   validation sample s. 
%
%   y - an S by 1 array.  y(s) gives the label for validation sample s.
%
%   dT - the base tree to prune a node from. 
% 
% Outputs: 
%
%   dTOut - the pruned tree. 
%
function dTOut = pruneSingleGreedyNode(x, y, dT)
    max = -inf;
    trees = pruneAllNodes(dT);
	for i=1:length(trees)
		a = getAccuracy(x, y, trees{i});
        if a > max
            max = a;
            dTOut = trees{i};
        end
    end
end

function A = getAccuracy(x, y, tree)
    predY = batchClassifyWithDT(x,tree);    %give a tree and test data and return the lables
    result = y - predY;
    nZeros = sum(result == 0);
    A = nZeros/size(y,1);
end