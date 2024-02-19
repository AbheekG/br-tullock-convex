"""
Implements the examples for non convergence for 2 non-homogeneous agents
"""

import numpy as np
import matplotlib.pyplot as plt


a = 1e-5  # the minimum output action

def br(y, c, r):
	# cost = c*z^r
	if y == 0: 
		return a
	else:
		if r == 1:
			return max(np.sqrt(y/c) - y, 0)
		elif r < 1:
			assert False, "r < 1 not supported."
			return 0
		else:
			# convex cost. binary search
			eps = 1e-15  # error in BR
			left = 0.0
			right = (1/c)**(1/r)

			max_steps = 100
			while max_steps > 0:
				x = (left + right)/2

				if y - c * r * x**(r-1) * (x + y)**2 > eps:
					left = x
				elif y - c * r * x**(r-1) * (x + y)**2 < -eps:
					right = x
				else:
					break

				max_steps -= 1
				if max_steps < 1:
					print(left, right)
			return x

def run_example(C, R, S):

	T = 10000
	history = np.zeros((T,2))  # store history
	# start state
	history[0,:] = S

	for t in range(T-1):
		# one agent moves
		if t % 2 == 0:
			# agent 0=2 plays
			history[t+1, 0] = br(history[t, 1], C[0], R[0])
			history[t+1, 1] = history[t, 1]
		else:
			# agent 1 plays
			history[t+1, 0] = history[t, 0]
			history[t+1, 1] = br(history[t, 0], C[1], R[1])

		# print(np.linalg.norm(history[t+1] - history[t], np.inf))
		if np.linalg.norm(history[t+1] - history[t], np.inf) < 1e-10:
			print(f"Converged after t={t} steps at {history[t+1]}")
			break


	# np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
	# print(history.transpose())
	for i in range(10): print("$%0.10f$" % history[T-10+i, 0], end=" & ")
	print()
	for i in range(10): print("$%0.10f$" % history[T-10+i, 1], end=" & ")
	print()
	# for i in range(11): print("c | " % history[i, 1], end="")


if __name__ == '__main__':
	
	# Example with c1(z) = z; c2(z) = z/10
	C = [1, 1/10]
	R = [1, 1]
	run_example(C, R, [0.1, 1])

	# Example with c1(z) = z^1.1; c2(z) = z^1.1/20
	C = [1, 1/20]
	R = [1.1, 1.1]
	run_example(C, R, [0.1058, 1.3102])
