using MacroModelling

@model Baxter_King_1993 begin
	uc[0] = c[0] ^ (-1)

	ul[0] = theta__l * l[0] ^ (-1)

	y[0] = A * k[-1] ^ theta__k * n[0] ^ theta__n

	fk[0] = n[0] ^ theta__n * A * theta__k * k[-1] ^ (theta__k - 1)

	fn[0] = k[-1] ^ theta__k * A * theta__n * n[0] ^ (theta__n - 1)

	gamma__x * k[0] = k[-1] * (1 - delta__k) + iv[0]

	l[0] + n[0] = 1

	c[0] + iv[0] = y[0] * (1 - tau) + tr[0] + check_walras[0]

	c[0] + iv[0] + gb[0] = y[0]

	y[0] * tau = tr[0] + gb[0]

	uc[0] = lambda[0]

	ul[0] = fn[0] * (1 - tau) * lambda[0]

	beta * lambda[1] * (1 + q[1] - delta__k) = gamma__x * lambda[0]

	q[0] = fk[0] * (1 - tau)

	gb[0] = GB_BAR + e_gb[x]

	1 + r[0] = gamma__x * lambda[0] / (beta * lambda[1])

	w[0] = fn[0]

end


@parameters Baxter_King_1993 begin
	A = 1.0

	gamma__x = 1.016

	theta__k = 0.42

	delta__k = 0.1

	N	=	0.2

	R	=	0.065

	sG	=	0.2

	tau_BAR	=	0.2

	tau = 0.2

	theta__n = 1-theta__k

	L = 1 - N

	beta = gamma__x/(1+R)

	Q = (gamma__x / beta - 1) + delta__k

	FK = Q / (1 - tau_BAR)

	K = (FK / (theta__k * A * N ^ theta__n)) ^ (1 / (theta__k - 1))

	FN = theta__n * A * K ^ theta__k * N ^ (theta__n - 1)

	IV = ((gamma__x - 1) + delta__k) * K

	Y = A * N ^ (1 - theta__k) * K ^ theta__k

	GB_BAR = sG*Y

	C = (Y - IV) - GB_BAR

	UC = C ^ -1

	UL = UC * (1 - tau_BAR) * FN

	theta__l = UL*L

end

