import numpy as np

# Romberg Integration Functions
def R00(a, b):
    return (1./2.) * (b-a) * (f(a) + f(b))

def RN0(Rn_minus_1, n, a, b):
    hn = (b-a)/(2**n)
    summation = sum(hn * f(a + (2*k-1)*hn) for k in range(1, 2**(n-1)+1))
    return (1./2.) * Rn_minus_1 + summation

def RNM(Rn_mm1, Rnm1_mm1, m):
    return (1./(4**m - 1.)) * ((4**m) * Rn_mm1 - Rnm1_mm1)

def R_nm(n, m, a, b):
    RR = np.zeros((21, 21))
    RR[0][0] = R00(a, b)
    for N in range(1, n+1):
        RR[N][0] = RN0(RR[N-1][0], N, a, b)
    for M in range(1, m+1):
        for N in range(M, n+1):
            RR[N][M] = RNM(RR[N][M-1], RR[N-1][M-1], M)

    return RR

# Function for f(x)
def f(x):
    return np.sin(x)

def main_function():
    results = []
    for m in range(3):
        for n in range(m, 21):
            RR = R_nm(n, m, 0, 11*np.pi)
            results.append(RR)
    return results

results = main_function()

# Output or further processing of the results can be added here.
