% This is a function to train a binary decision tree.  Feature values can 
% be either real valued or discrete. 
%
% Usage: node = trainDT(x, y)
%
% Inputs: 
%
%   x - a S by D matrix, where S is the number of samples and D is the
%   length of a feature vector.   x(s,:) gives the feature vector for 
%   sample s. 
%
%   y - an S by 1 array.  y(s) gives the label for sample s. 
%
% Outputs: 
%
%   node - a structure containing the root node of the decision tree.
%   The decision tree  returned by this function is simply a nested set of
%   nodes.  Nodes can be one of two types: a leaf node OR the root of
%   another tree.  
%
%   If a node is a leaf node, it will be a structure with the following
%   fields: 
%
%       isLeaf - a field with value true, stating the node is a leaf node
%
%       label - the label for samples which follow the tree down to this
%       leaf node. 
%
%       trainData - 2 by 1 array.  trainData(1) gives the label for 
%       this leaf node and trainData(2) gives the number of training
%       samples that were classified according to the path for this leaf. 
%
%   If a node is not a leaf node, it will be the root of another tree, and
%   will have the following fields: 
%
%       isLeaf - a field with value false, stating the node is not a leaf
%       node and simply the root of another tree
%
%       attr - the index of the attribute of the feature vector to split on
%       for this node. 
%
%       thresh - the threshold value to split on for this node. 
%
%       child1 - a structure for the root node used to further classify the feature
%       vector for a sample, x, when x(attr) <= thresh
%
%       child2 - a structure for the root node used to further classify the
%       feature vector for a sample, x, when x(attr) > thresh. 
%
%       trainData - a 2 row matrix.  The top row will list the labels for
%       all training data points that were classified under this node.  The
%       bottom row will list the number of training points for each label
%       that were classified under this node. 
%
function node = trainDT(x, y)

% Make sure we have consistent training data
nSmps = size(x,1); 
for s = 1:nSmps
    matchRows = all(repmat(x(s,:), [nSmps, 1]) == x, 2);
    matchedLabels = y(matchRows);
    assert(all(matchedLabels == matchedLabels(1)), 'Training data is not consistent.'); 
end

% Determine if we have a pure node
pureNode = all(y == y(1)); 

if pureNode
    node.isLeaf = true; 
    node.label = y(1); 
    node.trainData = [y(1); length(y)]; 
else
    
    % Find optimal split for the data
    [attr, thresh] = computeOptimalSplit(x, y); 
    
    indsLessThanOrEqual = (x(:, attr) <= thresh); 
    indsAbove = (x(:, attr) > thresh); 
    
    node.isLeaf = false; 
    node.attr = attr;
    node.thresh = thresh; 
    node.child1 = trainDT(x(indsLessThanOrEqual,:), y(indsLessThanOrEqual));
    node.child2 = trainDT(x(indsAbove,:), y(indsAbove));
    
    % Store information on training data for this root
    uniqueLabels = unique(y); 
    node.trainData = [uniqueLabels'; histc(y, uniqueLabels)']; 
end