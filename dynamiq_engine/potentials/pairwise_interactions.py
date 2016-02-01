import numpy as np


class PairwiseInteraction(object):
    def f(self, x):
        raise NotImplementedError()

    def dfdx(self, x):
        raise NotImplementedError()

    def d2fdx2(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.f(x)

    # these are only relevant if you have atom-atom distances to calculate
    def pdot_part(self, snapshot, i, j, pdot):
        pass #pragma: no cover

    def V_part(self, snapshot, i, j):
        pass #pragma: no cover

class ConstantInteraction(PairwiseInteraction):
    def __init__(self, value):
        self.value = value

    def f(self, x):
        return self.value

    def dfdx(self, x):
        return 0.0

    def d2fdx2(self, x):
        return 0.0

class HarmonicOscillatorInteraction(PairwiseInteraction):
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0

    def f(self, x):
        return 0.5*self.k*(x-self.x0)*(x-self.x0)

    def dfdx(self, x):
        return self.k*(x-self.x0)

    def d2fdx2(self, x):
        return self.k


class TanhInteraction(PairwiseInteraction):
    def __init__(self, a, V0, R0=0.0):
        self.a = a
        self.V0 = V0
        self.R0 = R0
        # TODO: might be good to have a way to set position-dependent var
        # (tanh_aR and sech_aR) as a first step of the integrator, avoiding
        # repeated computation. Requires careful usage within the integrator

    def f(self, x):
        tanh_aR = np.tanh(self.a * (x-self.R0))
        return self.V0 * tanh_aR

    def dfdx(self, x):
        sech_aR = 1.0 / np.cosh(self.a * (x - self.R0))
        return self.V0*self.a*sech_aR*sech_aR

    def d2fdx2(self, x):
        sech_aR = 1.0 / np.cosh(self.a * (x - self.R0))
        tanh_aR = np.tanh(self.a * (x-self.R0))
        return -2.0 * self.V0 * self.a * self.a * sech_aR * sech_aR * tanh_aR


class MorseInteraction(PairwiseInteraction):
    def __init__(self, D, beta, x0):
        self.D = D
        self.beta = beta
        self.x0 = x0

    def f(self, x):
        exp_mbetadx = np.exp(-self.beta * (x - self.x0))
        return self.D * (1.0 - exp_mbetadx) * (1.0 - exp_mbetadx)

    def dfdx(self, x):
        exp_mbetadx = np.exp(-self.beta * (x - self.x0))
        two_Db = 2*self.D*self.beta
        return two_Db * (exp_mbetadx - exp_mbetadx * exp_mbetadx)

    def d2fdx2(self, x):
        exp_mbetadx = np.exp(-self.beta * (x - self.x0))
        two_Db = 2*self.D*self.beta
        return two_Db*self.beta * (2*exp_mbetadx*exp_mbetadx - exp_mbetadx)


class QuarticInteraction(PairwiseInteraction):
    def __init__(self, A, B, C, D, E0, x0):
        x0_2 = x0*x0
        x0_3 = x0_2 * x0
        x0_4 = x0_3 * x0
        self.alpha = A
        self.beta = -4*A*x0 + B
        self.gamma = 6*A*x0_2 - 3*B*x0 + C
        self.delta = -4*A*x0_3 +3*B*x0_2 - 2*C*x0 + D
        self.epsilon = A*x0_4 - B*x0_3 + C*x0_2 - D*x0 + E0

    def f(self, x):
        x2 = x*x
        return (self.alpha*x2*x2 + self.beta*x2*x + self.gamma*x2 +
                self.delta*x + self.epsilon)

    def dfdx(self, x):
        x2 = x*x
        return (4.0*self.alpha*x2*x + 3.0*self.beta*x2 + 2.0*self.gamma*x +
                self.delta)

    def d2fdx2(self, x):
        x2 = x*x
        return (12.0*self.alpha*x2 + 6.0*self.beta*x + 2.0*self.gamma)

class GaussianInteraction(PairwiseInteraction):
    def __init__(self, A, alpha, x0=0.0):
        self.A = A
        self.alpha = alpha
        self.x0 = x0

    def f(self, x):
        dx = x-self.x0
        func = self.A*np.exp(-self.alpha*dx*dx)
        return func

    def dfdx(self, x):
        dx = x-self.x0
        func = self.A*np.exp(-self.alpha*dx*dx)
        return -2.0*self.alpha*dx*func

    def d2fdx2(self, x):
        dx = x-self.x0
        func = self.A*np.exp(-self.alpha*dx*dx)
        return 2.0*self.alpha*(2.0*self.alpha*dx*dx - 1.0)*func

class LennardJonesInteraction(PairwiseInteraction):
    pass

class WCAInteraction(PairwiseInteraction):
    pass

class GeneralizedWCAInteraction(PairwiseInteraction):
    # arbitrary powers
    pass

