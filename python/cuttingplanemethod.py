#!/usr/bin/env python

import abc, cStringIO, itertools, logging, sys

import numpy as np
import cvxpy as cp
from scipy import optimize

from polynomialalgebra import getpolynomialndmonomials, minimizepolynomialnd, minimizequadratic, minimizequartic

logger = logging.getLogger("cuttingplanemethod")

class CuttingPlaneMethodBase(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, x0, sigma, maxfractionaladjustment=0, reportdeltafun=True, printlogaterror=True):
    if x0.shape != sigma.shape:
      raise ValueError("x0 and sigma have different shapes: {}, {}".format(x0.shape, sigma.shape))

    if len(x0.shape) == 1:
      x0 = np.array([[_] for _ in x0])
      sigma = np.array([[_] for _ in sigma])

    if len(x0) != self.xsize:
      raise ValueError("len(x0) should be {}, is actually {}".format(self.xsize, len(x0)))
    if len(sigma) != self.xsize:
      raise ValueError("len(sigma) should be {}, is actually {}".format(self.xsize, len(sigma)))

    self.__x0 = x0
    self.__sigma = sigma
    self.__constraints = []
    self.__results = None
    self.__maxfractionaladjustment = maxfractionaladjustment

    self.__reportdeltafun = reportdeltafun
    self.__funatminimum = 0

    self.__printlogaterror = printlogaterror
    if printlogaterror:
      self.__logstream = cStringIO.StringIO()
      self.__logstreamhandler = logging.StreamHandler(self.__logstream)
      logger.addHandler(self.__logstreamhandler)
      logger.setLevel(logging.INFO)

    x = self.__x = cp.Variable(self.xsize)

    shiftandscale_quadraticterm = shiftandscale_linearterm = shiftandscale_constantterm = 0
    for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
      #shiftandscale = (self.__x - x0column) / sigmacolumn
      #self.__loglikelihood += cp.quad_form(shiftandscale, np.diag([1]*self.xsize))
      shiftandscale_quadraticterm += np.diag(1 / sigmacolumn**2)
      shiftandscale_linearterm += -2 * x0column / sigmacolumn**2
      shiftandscale_constantterm += sum(x0column**2 / sigmacolumn**2)

    quadraticterm = cp.quad_form(x, shiftandscale_quadraticterm)
    linearterm = cp.matmul(shiftandscale_linearterm, x)
    constantterm = shiftandscale_constantterm
    self.__loglikelihood = quadraticterm + linearterm + constantterm

    self.__minimize = cp.Minimize(self.__loglikelihood)

    logger.info("x0:")
    logger.info(str(self.__x0))
    logger.info("sigma:")
    logger.info(str(self.__sigma))
    logger.info("quadratic coefficients:")
    logger.info(str(np.diag(shiftandscale_quadraticterm)))
    logger.info("linear coefficients:")
    logger.info(str(shiftandscale_linearterm))

  def __del__(self):
    if self.__printlogaterror:
      logger.handlers.remove(self.__logstreamhandler)

  @abc.abstractproperty
  def xsize(self): "can just be a class member"

  @abc.abstractmethod
  def evalconstraint(self, potentialsolution):
    """
    Evaluates the potential solution to see if it satisfies the constraints.
    Should return two things:
     - the minimum value of the polynomial that has to be always positive
     - the values of the monomials at that minimum
       e.g. for a 4D quartic, (1, x1, x2, x3, x4, x1^2, x1x2, ..., x4^4)
    """

  def iterate(self):
    if self.__results is not None:
      raise RuntimeError("Can't iterate, already finished")

    toprint = "starting iteration {}".format(len(self.__constraints)+1)
    logger.info("="*len(toprint))
    logger.info(toprint)
    logger.info("="*len(toprint))

    prob = cp.Problem(
      self.__minimize,
      self.__constraints,
    )

    solvekwargs = {
      "solver": cp.MOSEK,
    }
    try:
      prob.solve(**solvekwargs)
      x = self.__x.value

      if self.__reportdeltafun and not self.__constraints:
        self.__funatminimum = prob.value

      logger.info("found minimum {} at:\n{}".format(prob.value - self.__funatminimum, x))

      #does it satisfy the constraints?

      minimizepolynomial = self.evalconstraint(x)
      minvalue = minimizepolynomial.fun
    except BaseException as e:
      if self.__printlogaterror:
        print self.__logstream.getvalue()
      prob.solve(verbose=True, **solvekwargs)
      raise

    if minvalue >= 0:
      logger.info("Minimum of the constraint polynomial is %g --> finished successfully!", minvalue)
      self.__results = optimize.OptimizeResult(
        x=x,
        success=True,
        status=1,
        nit=len(self.__constraints)+1,
        maxcv=0,
        message="finished successfully",
        fun=prob.value - self.__funatminimum
      )
      return

    if -minvalue < x[0] * self.__maxfractionaladjustment:
      logger.info("Minimum of the constraint polynomial is %g", minvalue)

      oldx0 = x[0]
      multiplier = 1
      while minvalue < 0:
        lastx0 = x[0]
        print x[0], minvalue
        x[0] -= minvalue - multiplier*np.finfo(float).eps
        if x[0] == lastx0: multiplier += 1
        minvalue = self.evalconstraint(x).fun

      if x[0] / oldx0 - 1 < self.__maxfractionaladjustment:
        logger.info("Multiply constant term by (1+%g) --> new minimum of the constraint polynomial is %g", x[0] / oldx0 - 1, minvalue)
        logger.info("Approximate minimum of the target function is {} at {}".format(self.__loglikelihood.value - self.__funatminimum, x))
        self.__results = optimize.OptimizeResult(
          x=x,
          success=True,
          status=2,
          nit=len(self.__constraints)+1,
          maxcv=0,
          message="multiplied constant term by (1+{}) to get within constraint".format(x[0] / oldx0 - 1),
          fun=self.__loglikelihood.value - self.__funatminimum
        )
        return

    logger.info("Minimum of the constraint polynomial is {} at {} --> adding a new constraint using this minimum:\n{}".format(minvalue, minimizepolynomial.x, minimizepolynomial.linearconstraint))
    self.__constraints.append(
      cp.matmul(
        minimizepolynomial.linearconstraint[self.useconstraintindices,],
        self.__x
      ) >= np.finfo(float).eps
    )

  useconstraintindices = slice(None, None, None)

  def run(self, *args, **kwargs):
    while not self.__results: self.iterate(*args, **kwargs)
    return self.__results

class CuttingPlaneMethod1DQuadratic(CuttingPlaneMethodBase):
  xsize = 3
  evalconstraint = staticmethod(minimizequadratic)

class CuttingPlaneMethod1DQuartic(CuttingPlaneMethodBase):
  xsize = 5
  evalconstraint = staticmethod(minimizequartic)

class CuttingPlaneMethod4DQuadratic(CuttingPlaneMethodBase):
  xsize = 15
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(2, 4, coeffs)

class CuttingPlaneMethod4DQuartic(CuttingPlaneMethodBase):
  xsize = 70
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(4, 4, coeffs)

class CuttingPlaneMethod4DQuartic_4thVariableQuadratic(CuttingPlaneMethodBase):
  xsize = 65
  def insertzeroatindices():
    for idx, (coeff, variables) in enumerate(getpolynomialndmonomials(4, 4, [1]*70)):
      if variables["z"] >= 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())

  useconstraintindices = range(70)
  for _ in insertzeroatindices: useconstraintindices.remove(_)
  del _

  def evalconstraint(self, coeffs):
    coeffs = iter(coeffs)
    newcoeffs = np.array([0 if i in self.insertzeroatindices else next(coeffs) for i in xrange(70)])
    for remaining in coeffs: assert False
    return minimizepolynomialnd(4, 4, newcoeffs)

def cuttingplanemethod1dquadratic(*args, **kwargs):
  return CuttingPlaneMethod1DQuadratic(*args, **kwargs).run()
def cuttingplanemethod1dquartic(*args, **kwargs):
  return CuttingPlaneMethod1DQuartic(*args, **kwargs).run()
def cuttingplanemethod4dquadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquartic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablequadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableQuadratic(*args, **kwargs).run()

if __name__ == "__main__":
  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler(sys.stdout))
  a = np.array([[1, 2.]]*70)
  a[2,:] *= -1
  print CuttingPlaneMethod4DQuartic(
    a,
    abs(a),
    maxfractionaladjustment=1e-6,
  ).run()
