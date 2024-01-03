function [objfunctionval,predictionerror] = objfunction(thetavector,constants,thedata)
% USAGE: [objfunctionval,predictionerror] = objfunction(thetavector,constants,thedata)

% "objfunctionval" is a scalar which corresponds to the definition "empirical risk objective function" in problem.
% It is the empirical risk function whose loss function includes both the prediction error and hidden unit minimization term.

% "predictionerror" is a scalar which corresonds to the definition "prediction error" in problem.
% This is the same as "objfunctionval" when "constants.lambda = 0".


%--------------------------------BEGIN FUNCTION "getpredictedresponses"--------------------------
    function [predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata)
    % USAGE: [predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata)

    % Get Predicted Responses using "getpredictedresponses.m"
    % The ith column of "hidmx" is the activation pattern over the
    % H hidden units for the ith input pattern which is the
    % ith row of "inputvectors"  (see definition of "inputvectors" below)
    %
    % The ith column of the row vector "predictedresponses" is the activity
    % of the ith output unit given the ith input pattern which is the ith
    % row of input vectors (see definition of "inputvectors" below)

    % Unpack Constants
 
    nrhidden = constants.nrhidden;

    % Unpack Event History 
    eventhistory = thedata.eventhistory;
    nrtargets = thedata.nrtargets;
    [nrstim,nrvars] = size(eventhistory);

    % Get "desiredresponses". The ith row of "desired response"
    % for the ith row of "inputvectors". The matrices "desiredresponse"
    % and "inputvectors" comprise the "training data set"
    desiredresponse = eventhistory(:,(1:nrtargets));
    inputvectors = eventhistory(:,(nrtargets+1):nrvars);
    [nrstim,inputvectordim] = size(inputvectors);

    % Unpack parameter values
    % Let H be number of hidden units. Let each input vector have
    % dimension D. 
    % wmatrix: matrix with H rows and D columns as defined in problem
    % vmatrix: matrix with 1 rows and H+1 columns as defined in problem
    [wmatrix,vmatrix] = unpackparameters(thetavector,constants,thedata);

    % STUDENT: Now Compute Responses to Hidden Units 
    hidmx = logisticsigmoid(inputvectors * wmatrix');
    biased_hidmx = [hidmx, ones(size(hidmx, 1), 1)];
    
    % STUDENT: Now compute output unit responses (i.e., predicted responses)
     predictedresponses = logisticsigmoid(biased_hidmx * vmatrix');
    predictedresponses = predictedresponses'; 

    hidmx = hidmx';
    
    
    
    end  % end of "getpredictedresponses" function
% ----------------------------- END OF FUNCTION "getpredictedresponses"-------


% Unpack Constants
lambda = constants.lambda;
nrhidden = constants.nrhidden;

% Unpack Event History (same as "gradobjfunction.m")
eventhistory = thedata.eventhistory;
nrtargets = thedata.nrtargets;
[nrstim,nrvars] = size(eventhistory);

% Get "desiredresponses". The ith row of "desired response"
% for the ith row of "inputvectors". The matrices "desiredresponse"
% and "inputvectors" comprise the "training data set"
desiredresponse = eventhistory(:,(1:nrtargets));
inputvectors = eventhistory(:,(nrtargets+1):nrvars);
[nrstim,inputvectordim] = size(inputvectors);

% Unpack parameter values
% Let H be number of hidden units. Let each input vector have
% dimension D. 
% wmatrix: matrix with H rows and D columns as defined in problem
% vmatrix: matrix with 1 rows and H+1 columns as defined in problem
[wmatrix,vmatrix] = unpackparameters(thetavector,constants,thedata);

% Now Compute Responses to Hidden Units (same as "gradobjfunction.m")
% Get Predicted Responses using "getpredictedresponses.m"
% The ith column of "hidmx" is the activation pattern over the
% H hidden units for the ith input pattern which is the
% ith row of "inputvectors"
%
% The ith column of the row vector "predictedresponses" is the activity
% of the ith output unit given the ith input pattern which is the ith
% row of input vectors
[predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata);

% STUDENT: Compute "predictionerror" 
% evaluated at "thetavector" as defined in the problem.
% Define a small constant epsilon for numerical stability
epsilon = 1e-10;

% Compute Prediction Error using Binary Cross-Entropy
predictionerror = 0;

for i = 1:nrstim
    d_i = desiredresponse(i); 
    y_i = predictedresponses(i);

    % Clamping y_i to avoid extreme values
    y_i = max(min(y_i, 1 - epsilon), epsilon);

    % Binary Cross-Entropy for ith record
    error_i = -(d_i * log(y_i) + (1 - d_i) * log(1 - y_i));

    % Accumulate error over all records
    predictionerror = predictionerror + error_i;
end
% Average prediction error over all records
predictionerror = predictionerror / nrstim;

% STUDENT: Compute "objfunctionval" 
% evaluated at "thetavector" as defined in the problem.
objfunctionval = 0;

% Compute Objective Function Value
for i = 1:nrstim
    d_i = desiredresponse(i);  % Desired response for ith record
    y_i = predictedresponses(i);  % Predicted response for ith record
    h_i = hidmx(:, i);  % Hidden unit activations for ith record

    % Prediction error for ith record using Binary Cross-Entropy
    y_i = max(min(y_i, 1 - epsilon), epsilon);  % Clamping y_i
    error_i = -(d_i * log(y_i) + (1 - d_i) * log(1 - y_i));

    % Loss function for ith record (including prediction error and regularization)
    loss_i = error_i + (lambda / 2) * norm(h_i)^2;

    % Accumulate loss over all records
    objfunctionval = objfunctionval + loss_i;
end

% Average over all records
objfunctionval = objfunctionval / nrstim;

end  % END MAIN FUNCTION
