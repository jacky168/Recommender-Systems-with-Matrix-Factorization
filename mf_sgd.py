import numpy as np
import math
'''
R: no_user x no_item matrix
no_feature: number of latent factors
alpha: regularization constant controlling the strength of the reqularization termios
gamma: learning rate
maxiter: maximum iterations
'''
def sgdTraining(R, no_feature, alpha, gamma = 0.01, maxiter = 1000):
	np.random.seed(0)
	no_user, no_item = R.shape

	# each row of P matrix represents a user with no_feature latent factors
	P = np.random.rand(no_user, no_feature)

	# each row of Q matrix represents an item with no_feature latent factors
	Q = np.random.rand(no_item, no_feature)

	for _ in xrange(maxiter):
		for u in xrange(no_user):
			for i in xrange(no_item):
				if R[u,i] != 0:
					e = R[u,i] - np.dot(P[u].T, Q[i])
					Q[i] = Q[i] + gamma * (e*P[u] - alpha*Q[i])
					P[u] = P[u] + gamma * (e*Q[i] - alpha*P[u])

	return P, Q

if __name__ == "__main__":
    R = [
         [5,2,0,1,0,2],
         [4,0,0,0,2,0],
         [1,1,0,5,2,3],
         [1,0,0,4,0,5],
         [0,1,5,4,3,0],
        ]

    R = np.array(R)
    no_feature = 2
    alpha = 0.1
    P, Q = sgdTraining(R, no_feature, alpha)

    print("P matrix is:" + str(P))

    print("Q matrix is:" + str(Q))

    print("the original matrix is:")
    print(R)

    print("the product of P and Q is: ")
    print(np.around(np.dot(P, Q.T),decimals=2))

