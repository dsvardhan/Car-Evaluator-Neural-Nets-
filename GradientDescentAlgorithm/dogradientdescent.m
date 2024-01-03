function thetavector = dogradientdescent(constants,trainingdata);
% USAGE: thetavector = dogradientdescent(constants,trainingdata);

% Set Default Constants based upon local
% data file "constantvalues.mat"
initialwtmag = constants.initialwtmag;
nrhidden = constants.nrhidden;
learningrate = constants.learningrate;
maxgradnorm = constants.maxgradnorm;
maxiterations = constants.maxiterations;
gradfilename = constants.gradfilename;
errorfilename = constants.errorfilename;
displayfrequency = constants.displayfrequency;
temperature = constants.temperature;
displayfrequency = constants.displayfrequency;

% Setup training data and testing data constants
nrtargets = trainingdata.nrtargets;
inputvectordim = trainingdata.inputvectordim;
nrvars = trainingdata.nrvars;
inputvectors = trainingdata.inputvectors;
targetvectors = trainingdata.targetvectors;
varnames = trainingdata.varnames;
[nrtrainingstimuli,nrvars] = size(trainingdata.eventhistory);

% Initialize Parameters to Random Values
vmatrix = initialwtmag * randn(nrtargets,nrhidden+1);
wmatrix = initialwtmag * randn(nrhidden,(nrvars-nrtargets));
wmatrixT = wmatrix';
vmatrixT = vmatrix';
initialwtvalues = [vmatrixT(:); wmatrixT(:)];

% Start Descent Algorithm
errorhistory = []; gradnormhistory = [];
keepgoing = 1; thetavector = initialwtvalues;
iteration = 0;
while keepgoing,
    % This function evaluates the function containing the gradient
    % of the objective function which is located in the file
    % "gradfilename" evaluated at thetavector, constants, trainingdata
    % 
    %  thetavector = column vector whose elements are THETA
    %  constants = data structure containing key constants
    %  trainingdata = data structure containing the "training data"
    thegradient = feval(gradfilename,thetavector,constants,trainingdata);

    % This function evaluates the function containing 
    % the objective function which is located in the file
    % "objfilename" evaluated at thetavector, constants, trainingdata 
    [objfunkval,predicterror] = feval(errorfilename,thetavector,constants,trainingdata);

    % Update the parameter estimates using the batch gradient descent rule
    thetavector = parameterupdate(thetavector,thegradient,learningrate);
   
    % Compute the "keepgoing" flag to determine whether or not to stop
    [keepgoing,gradinfnorm] = computekeepgoingcondition(thegradient,iteration,maxiterations,maxgradnorm);

    % Display Results During the Learning Process
    if mod(iteration,displayfrequency)  == 0,
        disp(['Iteration #',num2str(iteration),', Objective Function Value = ',num2str(objfunkval),...
              ', Prediction Error = ',num2str(predicterror),', Grad Infinity Norm = ', num2str(gradinfnorm)]);
    end;
    iteration = iteration + 1;
end;
disp(['------Iterations Completed------.'])

