import numpy as np

def EM_step(x,dt=1,mu=0,sigma=1,theta=1, **kwargs):
    x = np.matrix(x)
    Z = np.random.normal(size=x.shape)
    return x + theta*(mu-x)*dt + np.sqrt(2*dt)*sigma*Z

def stationary(size=1, mu=0, sigma=1, theta=1, **kwargs):
    return np.random.normal(loc = mu, scale = sigma/np.sqrt(theta), size=size)

# N - # slices of the interval [0,1]
def trajectory(x0,T,N=1,**kwargs):
    X=np.matrix(x0).T
    x,y=x0,x0
    for i in range(T*N):
        x, y = y, EM_step(x, dt=1/N, **kwargs)
        if i%N == N-1:
            X = np.hstack([X,y.T])
    return X

def gen_data(path, **kwargs):
    x0 = stationary(**params)
    X = trajectory(x0, **params)
    np.save(path, X)

if __name__ == '__main__':
    params = {
        'N': 10,
        'T': 100,
        'size': 10000,
        'mu': 1.,
        'sigma': 0.2,
        'theta': 1.,
    }
    gen_data('data/train0', **params)

    params['size'] = 1000
    gen_data('data/test', **params)

    params['size'] = 10000
    params.update({
        'mu': np.random.uniform(low=-5,high=5, size=params['size']),
        'sigma': np.random.beta(a=2,b=8,size=params['size']),
    })
    gen_data('data/train1', **params)
