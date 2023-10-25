function [W, b, f, f_init] = PVCC_multi(data, truth, options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize
c = options.c;
v = options.v;
SS = options.SS;
comP = options.comP;
labelP = options.labelP;
lambda_1 = options.lambda_1;
lambda_2 = options.lambda_2;

maxIterPVCC = 150;
epsilon = 1e-6;
rand('seed', 0);

objValue = zeros(1, maxIterPVCC);

H = cell(1, v);

C = cell(1, v);
A = cell(1, v);
B = cell(1, v);
s = cell(1, v);

N = size(data{1}, 1);
numL = floor(N*labelP);
numC = floor(N*comP);
full_l = randperm(N);
l_index = full_l(1:numL);

for iterV = 1:v
	dim(iterV) = size(data{iterV}, 2);

	eta(iterV) = numC;
	
	X{iterV} = zeros(N, dim(iterV));
	X{iterV}(1:numC, :) = data{iterV}(1:numC, :);
	
	P{iterV} = zeros(N, c);
	P{iterV}(1:numC, :) = ones(numC, c);

	E{iterV} = zeros(N, 1);
	E{iterV}(1:numC, :) = ones(numC, 1);

	Gamma{iterV} = [1:numC];
end


for iIns = numC+1:N
	randV = floor(v*rand()) + 1;

	eta(randV) = eta(randV) + 1;
	X{randV}(iIns, :) = data{randV}(iIns, :);
	P{randV}(iIns, :) = ones(1, c);
	E{randV}(iIns) = 1;
	Gamma{randV} = [Gamma{randV}, iIns];
end

for iterV = 1:v
	C{iterV} = eye(N) - E{iterV}*ones(1,N)/eta(iterV);
	B{iterV} = X{iterV}' * C{iterV}';
	A{iterV} = B{iterV}*B{iterV}' + eta(iterV)*lambda_1*eye(dim(iterV));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the view-specific similarity matrix S_i and global 
% similarity matrix M
M = zeros(N, N);
cntS = zeros(N, N);
for iterV = 1:v
	s{iterV} = zeros(N, N);
	tmpX{iterV} = zeros(N, dim(iterV));
	for iInst = 1:N
		if norm(X{iterV}(iInst, :)) ~= 0
			tmpX{iterV}(iInst, :) = X{iterV}(iInst, :)/norm(X{iterV}(iInst, :));
		else
			tmpX{iterV}(iInst, :) = zeros(1, dim(iterV));
		end
	end
	tmpS = tmpX{iterV}*tmpX{iterV}';
	s{iterV}(Gamma{iterV}, Gamma{iterV}) = tmpS(Gamma{iterV}, Gamma{iterV});
	cntS(Gamma{iterV}, Gamma{iterV}) = cntS(Gamma{iterV}, Gamma{iterV}) + 1;
	M = M + s{iterV};
    s{iterV}(Gamma{iterV}, Gamma{iterV});
end

M(find(cntS>0)) = M(find(cntS>0)) ./ cntS(find(cntS>0));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimize the target function

% Initialize two variables	
f = PVCC_init(truth, l_index, M, 3);
f_init = f;

% Calulate the H matrix
sumH = zeros(N, N);
for i = 1:v
	aMb{i} = A{i}\B{i};
	H{i} = C{i}*C{i}'*aMb{i}'*(lambda_1/2*aMb{i}+1/2/eta(i)*B{i}*B{i}'*aMb{i}-1/eta(i)*B{i}) + 1/2/eta(i)*C{i}'*C{i};
	% Set the columns and rows of H to zeros
	H{i} = PVCC_Omega(H{i}, Gamma{i});
	sumH = sumH + H{i};
end

% Optimize f
f_old = f;
for iter1 = 1:maxIterPVCC
	% fprintf('---The %d-th Outer iteration---\n', iter1);

	% Accelerated Gradient Descent
	% t_old = 1;
	% t_cur = 1;
	% f_next = f;
	% f_old = f;
	% grad_1 = sumH * f_next;
	% M_approx = f_next*f_next';
	grad_1 = sumH * f;
	M_approx = f*f';
	grad_2 = zeros(N, c);
	for iterV = 1:v
		dif{iterV} = M_approx - s{iterV};
		proj{iterV} = PVCC_Omega(dif{iterV}, Gamma{iterV});
		normV(iterV) = norm(proj{iterV}, 'fro');
		if normV(iterV) == 0
			proj{iterV} = 0;
		else
			proj{iterV} = proj{iterV} / normV(iterV);
        end
        size(proj{iterV}*f);
		grad_2 = grad_2 + proj{iterV}*f/eta(iterV);

	end
	grad_f = grad_1 + grad_2*lambda_2;

	f = f - SS*grad_f/iter1;
	tmp_0 = zeros(size(f));
	tmp_1 = ones(size(f));

	f(find(f<=0)) = tmp_0(find(f<=0));
	f(find(f>=1)) = tmp_1(find(f>=1));
	f(l_index, :) = truth(l_index, :);
	f_old = f;

	% t_cur = (1+sqrt(1+4*t_old^2))/2;
	% f_next = f + (t_old-1)/t_cur*(f-f_old);
	% f_old = f;
	% t_old = t_cur;
	% f = f - 1;
	[loss1(iter1), loss2(iter1)] = PVCC_cal_objValue(sumH, f, s, lambda_2, eta, Gamma);
	objValue(iter1) = loss1(iter1) + loss2(iter1); 
	%l1 = loss1(iter1)
	l2 = loss2(iter1);
	% objValue(iter1);
	% if mod(iter1, 5) == 0
	% 	fprintf('	Current object value is %.4f\n', objValue(iter1));
	% end

	if iter1>1 && (abs(objValue(iter1)-objValue(iter1-1))/objValue(iter1)<epsilon)
		fprintf('   Object value converge to %.4f at iteration %d before the max OUTer Iteration reached\n', objValue(iter1), iter1);
		break;
	end

end

for i = 1:v
	W{i} = aMb{i}*C{i}*(f.*P{i});
	b{i} = ((f.*P{i}) - X{i}*W{i})'*ones(N, 1)/eta(i);
end


















