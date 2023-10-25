function [f] = PVCC_init(truth, label_index, M, K)


N = size(M, 1);
N_l = length(label_index);
c = size(truth, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize f_u using knn with M
f = zeros(N, c);
for i = 1:N
	if length(find(label_index == i)) > 0
		f(i, :) = truth(i, :);
		continue;
	end

	sumS = 0;
	[~, Ind] = sort(M(i, label_index), 'descend');
	res = zeros(1, c);

	for j = 1:K
		res = res + M(i, label_index(Ind(j)))*truth(label_index(Ind(j)), :);
		sumS = sumS + M(i, label_index(Ind(j)));
	end
	if (sumS ~= 0)
		f(i, :) = res / sumS;
	else
		f(i, :) = ones(1, c)/c;
	end

	% maxS = 0;
	% res = zeros(1, c);
	% for j = 1:length(label_index)
	% 	tmp = label_index(j);
	% 	if M(i, tmp) > maxS
	% 		maxS = M(i, tmp);
	% 		res = truth(tmp, :);
	% 	end
	% end
	% if maxS > 0
	% 	f(i, :) = res;
	% else
	% 	f(i, :) = ones(1, c)/c;
	% end

end







