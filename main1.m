function main1()
	trainX = csvread('trainX.csv');     %read csv file (train data)
	trainY = csvread('trainY.csv');
	tree = trainDT(trainX, trainY);     %create tree with train data

    state = gatherTreeStats(tree);
    fprintf('Number of All Nodes is %d and leaf Nodes is %d\n', state.nNodes, state.nLeafs);

	testX = csvread('testX.csv');
	testY = csvread('testY.csv');       %read csv file (test data)
    
    trainAccuracy = getAccuracy(trainX, trainY, tree);
    testAccuracy = getAccuracy(testX, testY, tree);
    fprintf('Accuracy on train data is %f and on test data is %f\n', trainAccuracy, testAccuracy);

    
    validX = csvread('validationX.csv');
	validY = csvread('validationY.csv');       %read csv file (validation data)
    n = state.nNodes;
    accracies = zeros(3,1);
    xAxis = zeros(1,1);
    i = 1;
	while true
        state = gatherTreeStats(tree);
        xAxis(i) = state.nLeafs;
        accracies(1, i) = getAccuracy(trainX, trainY, tree);
        accracies(2, i) = getAccuracy(validX, validY, tree);
        accracies(3, i) = getAccuracy(testX, testY, tree);
        if state.nNodes == 1
            break;
        end
        tree = pruneSingleGreedyNode(validX, validY, tree);
        i = i+1;
    end
    hold on;
    plot(xAxis,accracies(1,:), 'color', 'red');
    plot(xAxis,accracies(2,:), 'color', 'green');
    plot(xAxis,accracies(3,:), 'color', 'blue');
    hold off;
end

function a = getAccuracy(x, y, tree)
    predY = batchClassifyWithDT(x,tree);    %give a tree and data and return the lables
    result = y - predY;
    nZeros = sum(result == 0);
    a = nZeros/size(y,1);
end