function [predictedresponses,hidmx] = getpredictedresponse(thetavector,constants,thedata)
% USAGE: [predictedresponses,hidmx] = getpredictedresponse(thetavector,constants,thedata)

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
% vmatrix: matrix with 1 rows and H+1 columns as defined in problem (rmg 11/29/2023)
[wmatrix,vmatrix] = unpackparameters(thetavector,constants,thedata);

% STUDENT: Now Compute Responses to Hidden Units (same as "gradobjfunction.m")
% Compute Responses to Hidden Units
hidmx = logisticsigmoid(inputvectors * wmatrix');
 % No need to transpose hidmx here for the calculation of predictedresponses
    biased_hidmx = [hidmx, ones(size(hidmx, 1), 1)];

% STUDENT: Now compute output unit responses (i.e., predicted responses)
predictedresponses = logisticsigmoid(biased_hidmx * vmatrix');
predictedresponses = predictedresponses'; % Ensure it's a row vector

% Transpose hidmx to match the expected dimensions before returning
hidmx = hidmx';
