function [trainingerror,testingerror,TRAINpercentcorrect,TESTpercentcorrect] = runperceptron(lambda,nrhidden)
% USAGE: [trainingerror,testingerror,TESTpercentcorrect,TESTpercentcorrect] = runperceptron(lambda,nrhidden)

% Setup Constants
constants.initialwtmag = 0.0001;
constants.nrhidden = nrhidden;
constants.learningrate = 0.05;
constants.maxgradnorm = 0.005;
constants.maxiterations = 10000;
constants.gradfilename = 'gradobjfunction';
constants.errorfilename = 'objfunction';
constants.displayfrequency = 100;
constants.temperature = 1;
constants.lambda = lambda

% Load Data Set
datasetfilename = 'recodedcar1.xlsx';
nrtargets = 1;
thedata = makethedataset(datasetfilename,nrtargets);

% Randomly Permute the records in "thedata" data structure
nrrecords = thedata.nrrecords;
permutedrecordlocs = randperm(nrrecords);
nrtrainrecords = round(nrrecords/2);
trainingdatalocs = permutedrecordlocs(1:nrtrainrecords);
testingdatalocs = permutedrecordlocs((nrtrainrecords+1):nrrecords);

% Setup Training Data
trainingdata.datafilename = [datasetfilename,'TRAIN'];
trainingdata.eventhistory = thedata.eventhistory(trainingdatalocs,:);
trainingdata.varnames = thedata.varnames;
trainingdata.nrvars = thedata.nrvars;
trainingdata.inputvectordim = thedata.inputvectordim;
trainingdata.inputvectors = thedata.inputvectors(trainingdatalocs,:);
trainingdata.targetvectors = thedata.targetvectors(trainingdatalocs,:);
trainingdata.nrrecords = nrtrainrecords;
trainingdata.nrtargets = nrtargets;

% Setup Testing Data
testingdata.datafilename = [datasetfilename,'TEST'];
testingdata.eventhistory = thedata.eventhistory(testingdatalocs,:);
testingdata.varnames = thedata.varnames;
testingdata.nrvars = thedata.nrvars;
testingdata.inputvectordim = thedata.inputvectordim;
testingdata.inputvectors = thedata.inputvectors(testingdatalocs,:);
testingdata.targetvectors = thedata.targetvectors(testingdatalocs,:);
testingdata.nrrecords = nrrecords - nrtrainrecords;
testingdata.nrtargets = nrtargets;

% Run Gradient Descent Algorithm
parameterestimates = dogradientdescent(constants,trainingdata);

% Display final training and testing error
[trainobjfunkval,trainingerror] = objfunction(parameterestimates,constants,trainingdata);
trainpredictresponse = getpredictedresponses(parameterestimates,constants,trainingdata);
[testobjfunkval,testingerror] = objfunction(parameterestimates,constants,testingdata);
testpredictresponse = getpredictedresponses(parameterestimates,constants,testingdata);
traintargetvecs = trainingdata.targetvectors;
testtargetvecs = testingdata.targetvectors; 
trainoutput = trainpredictresponse > 0.5;
testoutput = testpredictresponse > 0.5;
TRAINpercentcorrect = mean(trainoutput(:) == traintargetvecs(:));
TESTpercentcorrect = mean(testoutput(:) == testtargetvecs(:));

disp(['LAMBDA = ',num2str(constants.lambda),', Number Hidden Units = ',num2str(constants.nrhidden)]);
disp(['Objective Function Training Error = ',num2str(trainobjfunkval),...
      ', Prediction Training Error = ',num2str(trainingerror),...
      ', % Correct (Train) = ',num2str(TRAINpercentcorrect*100),'%']);
disp(['Objective Function Testing Error = ',num2str(testobjfunkval),...
      ', Prediction Testing Error = ',num2str(testingerror),...
      ', % Correct (Test) = ',num2str(TESTpercentcorrect*100),'%']);
disp('==============================================================================================');
