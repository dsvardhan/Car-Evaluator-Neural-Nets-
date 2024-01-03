function [keepgoing,gradinfnorm] = computekeepgoingcondition(thegradientvector,iteration,maxiterations,maxgradnorm)

% keepgoing = 1 when iterations is strictly less than the maximum number of
% iterations "maxiterations" (scalar) and the infinity norm of the gradient is strictly greater than the 
% "maxgradnormval" (scalar). "OTHERWISE: set keepgoing = 0
%
% "gradinfnorm" is the infinity norm of "thegradientvector"

% STUDENT Compute these variables (scalars)
%gradinfnorm = ????
%keepgoing = ???
%keepgoing = 0;
gradinfnorm = max(abs(thegradientvector));
if iteration < maxiterations && gradinfnorm> maxgradnorm
    keepgoing=1;
else
    keepgoing=0;


end
