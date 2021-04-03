% Given a set of training data, this is a function to select the single
% best attribute to perform a binary split with, as measured by information
% gain, and the particular threshold to perform that split at. 
%
% Usage: [attr, thresh] = computeOptimalSplit(x, y)
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
%   attr - this is the index of the best attribute to perform a split at.
%   
%   thresh - this is the threshold at which the split should be performed.
%            
%   Thus, the splitting rule for sample s will be: 
%
%       split into class 1,      if x(s,attr) <= thresh, 
%       split into class 2,      otherwise
%       
% Example: 
%
%  If: 
%
%       x = [1 2 3 4 5 6 7 8 9; 2 2 2 2 2 2 2 2 2]'
% 
%       y = [0 0 0 0 0 1 1 1 1]'
%
%
%   and we call: 
%
%      [attr, thresh] = computeOptimalSplit(x, y)
%
%   then: 
%
%       attr = 1
%       
%       thresh = 5.5
%
function [attr, thresh] = computeOptimalSplit(x, y) %compute best attribute by information gain
% Please provide the code for this function 
    max = -inf;
    attr = -1;
    for i=1:size(x,2)
        [gain, th] = computeGainAndThreshold(x(:,i),y);
        if gain > max
            max = gain;
            attr = i;
            thresh = th;
        end
    end
end

function [gain, th] = computeGainAndThreshold(x,y) % x is a S by 1 matrix  and y is their labels (thus its dimensions is like x )
    if all(x == x(1))
        gain = 0;
        th = x(1);
    else
		n = size(x,1);
		z = [x,y];
		z = sortrows(z);     %sort z's rows by first column (increamental)
		e = Entropy(y);
		min = inf;
		imin = -1;
		for i=1:n-1
			if z(i,2) ~= z(i+1,2) %if lable z(i,2) is not equal to z(i+1,2) , we'll compute entropy of two splitted classes.
				I = (Entropy(y(1:i))*i/n) + (Entropy(y(i+1:n))*(n-i)/n);
				if I < min
					min = I;
					imin = i;
				end
			end
		end
		gain = e - min;
		th = (z(imin,1) + z(imin+1,1))/2;
    end
end

function e = Entropy(x)
    if all(x == x(1))
        e = 0;
    else
        n = size(x,1);
		nZeros = sum(x == 0);
		p1 = nZeros/n;
		p2 = (n - nZeros)/n;
        t1 = p1*log2(p1);
        t2 = p2*log2(p2);
		e = -(t1 + t2);
    end
end