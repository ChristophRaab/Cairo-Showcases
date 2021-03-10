function [Sim] = DoubleCentering(Dis)
	n=size(Dis,1);
	J = eye(n) - repmat(1/n,n,n);
	Sim = -0.5 * J * Dis * J;
	Sim=(Sim+Sim')/2;

