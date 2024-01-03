function thetanew = parameterupdate(thetacurrent,gradientvector,learningrate)

% This function combines the current parameter estimates 
% "thetacurrent" (vector)
% with the "gradient vector" (vector) and the "learning rate" (scalar) 
% to get the new parameter estimates "thetanew" using the 
% standard batch gradient descent algorithm formula 

% STUDENT Compute "thetanew" (vector)
%thetanew = ????
thetanew=thetacurrent-learningrate*gradientvector

end

