function prob = logisticsigmoid(netinput)
% USAGE: prob = logisticsigmoidnetinput)

prob = 1 ./ (1 + exp(-netinput));


end
