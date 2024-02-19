import numpy as np
import matplotlib.pyplot as plt
import pickle


def br(X, i, a, r = 1):
	if r == 1:
		y = sum(X) - X[i]
		return np.maximum(np.sqrt(y) - y, 0) + a*(y == 0)
	elif r < 1:
		assert False, "r < 1 not supported"
	else:
		assert r > 1, "r > 1 expected"
		y = sum(X) - X[i]
		if y == 0: 
			return a
		else:
			# convex cost. binary search
			eps = 1e-10  # error in BR
			left = 0
			right = 1

			max_steps = 100
			while max_steps > 0:
				x = (left + right)/2
				if y - r * x**(r-1) * (x + y)**2 > eps:
					left = x
				elif y - r * x**(r-1) * (x + y)**2 < -eps:
					right = x
				else:
					break

				max_steps -= 1
				if max_steps < 1:
					print(left, right)

			return x


def u(xi, X, i, r = 1):
	return xi / (xi + sum(X) - X[i]) - xi**r


def convergence_time(n, eps, a, start, select):
	"""
	n: number of agents
	eps: approximate equilibrium parameter
	a: smallest output
	"""
	print(n, eps, a, start, select)

	X_prev = -np.ones(n)  # variable to store the backup profile for time t-1

	# Start Profile Choices
	if start == "zero":
		# (A) start at (a, a, 0, ...)
		X = np.zeros(n)
		X[0] = a; X[1] = a
	elif start == "random":
		# (B) random start profile that sums up to < 1
		X = np.random.uniform(size=n)
		X = np.random.uniform() * X / X.sum()

	time_taken = 0  # convergence time
	i_t = 0
	history = []

	while True:
		# compute the max change in utility possible
		errors = np.zeros(n)
		s = np.sum(X)
		for j in range(n):
			errors[j] = u(br(X, j, a), X, j) - u(X[j], X, j)
		if np.linalg.norm(errors, np.inf) <= eps:
			break

		time_taken += 1

		# copy prev data
		X_prev = X.copy()
		i_prev = i_t

		# select the agent who takes the action
		if select == "unif":
			i_t = np.random.randint(n)
		elif select == "round":
			i_t = (i_prev + 1) % n
		elif select == "lex":
			first_above_eps = (errors > eps)
			i_t = first_above_eps.argmax()
		elif select == "worst":
			min_above_eps = errors + 10 * (errors <= eps)
			i_t = min_above_eps.argmin()
		elif select == "best":
			min_above_eps = errors
			i_t = min_above_eps.argmax()
		elif select == "all":
			i_t = range(n)
		else:
			assert False, "Should not reach this point."

		# compute next state
		X[i_t] = br(X, i_t, a)

		# store history
		history.append( (time_taken, i_t, X) )
		# print(time_taken)
		# print(X)

	return time_taken


def time_vs_eps(n, a, n_samples, start, select, m):
	if select != "unif": n_samples = 1  # for deterministic, one sample enough
	
	eps_array = (1/5)**np.arange(1, m+1)
	time = np.zeros(m)

	for i_m in range(m):
		eps = eps_array[i_m]
		time_m = np.zeros(n_samples)
		for i_sample in range(n_samples):
			time_m[i_sample] = convergence_time(n, eps, a, start, select)
		time[i_m] = time_m.mean()

	return time, eps_array


def time_vs_n(eps, a, n_samples, start, select, m):
	if select != "unif": n_samples = 1  # for deterministic, one sample enough
	
	n_array = 2 + np.arange(m)
	time = np.zeros(m)

	for i_m in range(m):
		n = n_array[i_m]
		time_m = np.zeros(n_samples)
		for i_sample in range(n_samples):
			time_m[i_sample] = convergence_time(n, eps, a, start, select)
		time[i_m] = time_m.mean()

	return time, n_array


def time_vs_a(n, eps, n_samples, start, select, m):
	if select != "unif": n_samples = 1  # for deterministic, one sample enough

	a_array = (1/10)**np.arange(1, m+1)
	time = np.zeros(m)

	for i_m in range(m):
		a = a_array[i_m]
		time_m = np.zeros(n_samples)
		for i_sample in range(n_samples):
			time_m[i_sample] = convergence_time(n, eps, a, start, select)
		time[i_m] = time_m.mean()

	return time, a_array



def experiment_eps(n, a, start, selects, good_selects, n_samples):
	# run or load
	# with open('logs/experiment_eps.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	# 	time = data[0]; eps_array = data[1]
	time = dict()
	for select in selects:
		time[select], eps_array = time_vs_eps(n, a, n_samples, start, select, 20)
	# data = (time, eps_array)
	# with open('logs/experiment_eps.pkl', 'wb') as f:
	# 	pickle.dump(data, f)

	# for a better plot
	factors = dict(zip(good_selects, [200, 200, 400]))
	labels = dict()
	for select in selects:
		if select in good_selects:
			labels[select] = str(factors[select])+select
		else:
			labels[select] = select
	for select in good_selects: 
		time[select] *= factors[select]


	# 1/eps
	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)

	for select in selects:
		ax_top_left.plot((1/eps_array), time[select], label=labels[select])

	ax_top_left.set(xlabel='(1/eps)', ylabel='convergence time')
	ax_top_left.legend()
	plt.show()


	# log(1/eps)
	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)
	for select in selects:
		ax_top_left.plot(np.log(1/eps_array), time[select], label=labels[select])

	ax_top_left.set(xlabel='log(1/eps)', ylabel='convergence time')
	ax_top_left.legend()
	plt.show()


	# loglog(1/eps)
	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)
	for select in selects:
		ax_top_left.plot(np.log(np.log(1/eps_array)), time[select], label=labels[select])

	ax_top_left.set(xlabel='loglog(1/eps)', ylabel='convergence time')
	ax_top_left.legend()
	plt.show()


def experiment_n(eps, a, start, selects, good_selects, n_samples):

	# run or load
	# with open('logs/experiment_n.pkl', 'rb') as f:
		# data = pickle.load(f)
		# time = data[0]; n_array = data[1]
	time = dict()
	for select in selects:
		time[select], n_array = time_vs_n(eps, a, n_samples, start, select, m = 20)
	data = (time, n_array)
	# with open('logs/experiment_n.pkl', 'wb') as f:
	# 	pickle.dump(data, f)


	# for a better plot
	factors = dict(zip(good_selects, [100, 40, 200]))
	labels = dict()
	for select in selects:
		if select in good_selects:
			labels[select] = str(factors[select])+select
		else:
			labels[select] = select
	for select in good_selects:
		time[select] *= factors[select]


	# n
	fig = plt.figure()
	ax_top_right = fig.add_subplot(1, 1, 1)
	for select in selects:
		ax_top_right.plot(n_array * np.log(n_array), time[select], label=labels[select])
	ax_top_right.set(xlabel='n log(n)', ylabel='convergence time')
	ax_top_right.legend()
	plt.show()

	# n^2
	fig = plt.figure()
	ax_top_right = fig.add_subplot(1, 1, 1)
	for select in selects:
		ax_top_right.plot(n_array**2, time[select], label=labels[select])
	ax_top_right.set(xlabel='n^2', ylabel='convergence time')
	ax_top_right.legend()
	plt.show()

	# n^3
	fig = plt.figure()
	ax_top_right = fig.add_subplot(1, 1, 1)
	for select in selects:
		ax_top_right.plot(n_array**3, time[select], label=labels[select])
	ax_top_right.set(xlabel='n^3', ylabel='convergence time')
	ax_top_right.legend()
	plt.show()


def experiment_a(n, eps, start, selects, good_selects, n_samples):
	# run or load
	# with open('logs/experiment_a.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	# 	time = data[0]; a_array = data[1]
	time = dict()
	for select in selects:
		time[select], a_array = time_vs_a(n, eps, n_samples, start, select, 20)
	data = (time, a_array)
	# with open('logs/experiment_a.pkl', 'wb') as f:
	# 	pickle.dump(data, f)


	# for a better plot
	factors = dict(zip(good_selects, [20, 20, 20]))
	labels = dict()
	for select in selects:
		if select in good_selects:
			labels[select] = str(factors[select])+select
		else:
			labels[select] = select


	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)
	for select in selects:
		if select in good_selects: time[select] *= factors[select]
		ax_top_left.plot(np.log(np.log(1/a_array)), time[select], label=labels[select])
		if select in good_selects: time[select] /= factors[select]

	ax_top_left.set(xlabel='loglog(1/gamma)', ylabel='convergence time')
	ax_top_left.set_ylim([0,10000])
	ax_top_left.legend()
	plt.show()

	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)
	for select in selects:
		reference = time[select][0]
		if select in selects: time[select] -= reference
		ax_top_left.plot(np.log(np.log(1/a_array)), time[select], label=select)
		if select in selects: time[select] += reference

	ax_top_left.set(xlabel='loglog(1/gamma)', ylabel='relative convergence time')
	ax_top_left.legend()
	plt.show()


def main():	
	n_samples = 100  # number of samples
	n = 10  # number of agents
	eps = 1e-10
	a = 1e-10
	start = "zero"
	selects = ["unif", "round", "best", "lex", "worst"]
	good_selects = ["unif", "round", "best"]

	# experiment_eps(n, a, start, selects, good_selects, n_samples)
	experiment_n(eps, a, start, selects, good_selects, n_samples)
	experiment_a(n, eps, start, selects, good_selects, n_samples)


def agent_3_eps():
	n = 3  # number of agents
	a = 1e-10
	start="zero"
	n_samples = 100
	selects = ["unif", "round", "best"]

	# with open('logs/experiment_agent_3_eps.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	# 	time = data[0]; eps_array = data[1]
	time = dict()
	for select in selects:
		time[select], eps_array = time_vs_eps(n, a, n_samples, start, select, 20)
	data = (time, eps_array)
	# with open('logs/experiment_agent_3_eps.pkl', 'wb') as f:
	# 	pickle.dump(data, f)

	fig = plt.figure()
	ax_top_left = fig.add_subplot(1, 1, 1)
	for select in ["unif", "round", "best"]:
		ax_top_left.plot(np.log(1/eps_array), time[select], label=select)

	ax_top_left.set(xlabel='log(1/eps)', ylabel='convergence time')
	ax_top_left.legend()
	plt.show()


if __name__ == '__main__':
	main()
	# agent_3_eps()