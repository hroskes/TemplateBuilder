#!/usr/bin/env python

import abc, collections, cStringIO, itertools, logging, multiprocessing, random, sys, warnings

import numpy as np
import cvxpy as cp
from scipy import optimize, special

from optimizeresult import OptimizeResult
from polynomialalgebra import HomogeneousDoubleQuadratic, HomogeneousPolynomialNd, minimizequadratic, minimizequartic, PolynomialNd

class CuttingPlaneMethodBase(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, x0, sigma, maxfractionaladjustment=0, reportdeltafun=True, printaswego=True, printlogaterror=True, moreargsforevalconstraint=(), maxiter=float("inf"), logprefix=None):
    x0 = np.array(x0)
    sigma = np.array(sigma)

    if x0.shape != sigma.shape:
      raise ValueError("x0 and sigma have different shapes: {}, {}".format(x0.shape, sigma.shape))

    if len(x0.shape) == 1:
      x0 = np.array([[_] for _ in x0])
      sigma = np.array([[_] for _ in sigma])

    if len(x0) != self.xsize:
      raise ValueError("len(x0) should be {}, is actually {}".format(self.xsize, len(x0)))
    if len(sigma) != self.xsize:
      raise ValueError("len(sigma) should be {}, is actually {}".format(self.xsize, len(sigma)))

    self.__cuttingplanes = []
    self.__otherconstraints = []
    self.__moreargsforevalconstraint = moreargsforevalconstraint
    self.__results = None
    self.__maxfractionaladjustment = maxfractionaladjustment

    self.__reportdeltafun = reportdeltafun
    self.__funatminimum = 0

    self.__logger = logging.getLogger("cuttingplanemethod{}".format(random.randint(1, 10000)))
    self.__printlogaterror = printlogaterror
    self.__printaswego = printaswego

    if printaswego:
      self.__logstreamhandler = logging.StreamHandler()
      self.__logger.addHandler(self.__logstreamhandler)
      self.__logger.setLevel(logging.INFO)
    elif printlogaterror:
      self.__logstream = cStringIO.StringIO()
      self.__logstreamhandler = logging.StreamHandler(self.__logstream)
      self.__logger.addHandler(self.__logstreamhandler)
      self.__logger.setLevel(logging.INFO)

    if logprefix is not None:
      format = "\n" + logprefix.rstrip() + " - %(asctime)s"
      if multiprocessing.current_process().name != "MainProcess":
        format += " - "+multiprocessing.current_process().name
      format += ":\n\n%(message)s"
      formatter = logging.Formatter(format)
      self.__logstreamhandler.setFormatter(formatter)

    self.__initminimization(x0, sigma)

    self.__maxiter = maxiter

  def __initminimization(self, x0, sigma):
    x = self.__x = cp.Variable(self.xsize)
    self.__t = None

    self.__logger.info("x0:\n"+str(x0)+"\nsigma:\n"+str(sigma))

    #===================================================================
    #simplest approach:
    #for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
    #  shiftandscale = (self.__x - x0column) / sigmacolumn
    #  self.__tominimize += cp.quad_form(shiftandscale, np.diag([1]*self.xsize))
    #this is inefficient because the code doesn't treat a sum of quad_forms nicely
    #===================================================================

    #===================================================================
    #second approach:
    #shiftandscale_quadraticterm = shiftandscale_linearterm = shiftandscale_constantterm = 0
    #for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
    #  shiftandscale_quadraticterm += np.diag(1 / sigmacolumn**2)
    #  shiftandscale_linearterm += -2 * x0column / sigmacolumn**2
    #  shiftandscale_constantterm += sum(x0column**2 / sigmacolumn**2)

    #quadraticterm = cp.quad_form(x, shiftandscale_quadraticterm)
    #linearterm = cp.matmul(shiftandscale_linearterm, x)
    #constantterm = shiftandscale_constantterm
    #self.__tominimize = quadraticterm + linearterm + constantterm
    #===================================================================

    #===================================================================
    #third approach, starts out the same as the second with a change in notation
    Q = c = r = 0
    for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
      #note the 2 here! it's to get the 1/2 in 1/2 x^T Q x + c^T x + r
      Q += 2 * np.diag(1 / sigmacolumn**2)
      c += -2 * x0column / sigmacolumn**2
      r += sum(x0column**2 / sigmacolumn**2)
    #we are minimizing 1/2 x^T Q x + c^T x + r, as in https://docs.mosek.com/modeling-cookbook/cqo.html#equation-eq-cqo-qcqo

    #now we want to scale the coefficients to get numbers even closer to 1.
    #We could scale all the coefficients to have sigma=1.  This wouldn't help however,
    #because we would have to scale them back in order to evaluate the nd polynomial
    #and we'd get a linear constraint with big numbers, same as before.
    #Instead, we want to scale the _variables_ of the nd polynomial, or in other words
    #scale the coefficients according to how they multiply those variables.
    #This doesn't affect whether the polynomial ever goes negative.

    self.__logger.info("Q matrix:\n"+str(np.diag(Q))+"\nlinear coefficients:\n"+str(c))

    multiplyvariables, multiplycoeffs = self.__findmultiplycoeffs(np.diag(Q) ** .5)
    self.__multiplycoeffs = multiplycoeffs
    Q = np.diag(np.diag(Q) / multiplycoeffs**2)
    c /= multiplycoeffs

    self.__multipliedargsforevalconstraint = tuple(
      f(multiplyvariables, arg) for f, arg in itertools.izip(self.modifymoreargs(), self.__moreargsforevalconstraint)
    )

    self.__logger.info("Multiplied variables to get coefficients closer to 1 for minimization.\nQ matrix:\n"+str(np.diag(Q))+"\nlinear coefficients:\n"+str(c))

    t = self.__t = cp.Variable()
    self.__tominimize = 0.5 * cp.quad_form(x, Q) + cp.matmul(c, x)
    #===================================================================

    self.__minimize = cp.Minimize(self.__tominimize)

    self.__otherconstraints += [
      self.__x[i]>=np.finfo(float).eps for i in self.maxpowerindices
    ]

  @property
  def polynomialvariables(self):
    return sorted(sum(self.monomials, collections.Counter()).keys()) #including '1'

  @staticmethod
  def modifymoreargs():
    return ()

  @property
  def maxpowerindices(self):
    maxpowers = {
      varname: max(monomial[varname] for monomial in self.monomials)
        for varname in self.polynomialvariables
    }
    for v, p in maxpowers.iteritems():
      if p%2: raise ValueError("max power of {} is odd: {}".format(v, p))
    result = []
    for i, monomial in enumerate(self.monomials):
      if all(monomial[v] == p or monomial[v] == 0 for v, p in maxpowers.iteritems()):
        result.append(i)
    return result

  def __findmultiplycoeffs(self, diagF, verbose=False):
    monomials = self.monomials()

    logdiagF = np.log(diagF)
    logmultiplyvariables = collections.defaultdict(cp.Variable)

    tominimize = 0
    for monomial, logoneovercoeff in itertools.izip_longest(monomials, logdiagF):
      tominimize += (logoneovercoeff + sum(logmultiplyvariables[variable] for variable in monomial.elements()))**2
    minimize = cp.Minimize(tominimize)
    prob = cp.Problem(minimize)
    prob.solve(verbose=verbose)

    logmultiplycoeffs = -np.array([sum(logmultiplyvariables[variable].value for variable in monomial.elements()) for monomial in monomials])
    multiplyvariables = {k: np.exp(v.value) for k, v in logmultiplyvariables.iteritems()}
    multiplycoeffs = np.exp(logmultiplycoeffs)

    if verbose:
      print {k: v.value for k, v in logmultiplyvariables.iteritems()}
      print logdiagF
      for i, logcoeff in enumerate(logmultiplycoeffs):
        #subtract because diagF is proportional to 1/coeff
        logdiagF[i] -= sum(logmultiplyvariables[variable].value for variable in monomial.elements())
      print logdiagF
      print diagF
      print np.exp(logdiagF)

    return multiplyvariables, multiplycoeffs

  def __del__(self):
    if self.__printlogaterror or self.__printaswego:
      self.__logger.handlers.remove(self.__logstreamhandler)

  @abc.abstractproperty
  def xsize(self): "can just be a class member"
  @abc.abstractmethod
  def constantindex(self, minimizepolynomialresults):
    """
    can be a static function
    index of the constant term of the polynomial
    or None if there isn't one
    """

  @abc.abstractmethod
  def evalconstraint(self, potentialsolution):
    """
    Evaluates the potential solution to see if it satisfies the constraints.
    Should return an OptimizeResult that includes:
     - the minimum value of the polynomial that has to be always positive
     - the values of the monomials at that minimum
       e.g. for a 4D quartic, (1, x1, x2, x3, x4, x1^2, x1x2, ..., x4^4)
    """
  @abc.abstractproperty
  def polynomialclass(self): pass
  @abc.abstractmethod
  def polynomialargs(self, coeffs): pass

  @property
  def monomials(self):
    return self.polynomialclass(*self.polynomialargs()).monomialswithoutcoeffs

  def iterate(self):
    if self.__results is not None:
      raise RuntimeError("Can't iterate, already finished")

    toprint = "starting iteration {}".format(len(self.__cuttingplanes)+1)
    self.__logger.info("="*len(toprint)+"\n"+toprint+"\n"+"="*len(toprint))

    prob = cp.Problem(
      self.__minimize,
      self.__otherconstraints + self.__cuttingplanes,
    )

    solvekwargs = {
      "solver": cp.GUROBI,
    }
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        prob.solve(**solvekwargs)
      x = self.__x.value

      if self.__reportdeltafun and not self.__cuttingplanes:
        self.__funatminimum = prob.value

      self.__logger.info("found minimum {} at:\n{}".format(prob.value - self.__funatminimum, x))

      #does it satisfy the constraints?

      minimizepolynomial = self.evalconstraint(x, *self.__multipliedargsforevalconstraint)
      minvalue = minimizepolynomial.fun
    except BaseException as e:
      if self.__printlogaterror and not self.__printaswego:
        print self.__logstream.getvalue()
      prob.solve(verbose=True, **solvekwargs)
      raise

    if minvalue >= 0:
      self.__logger.info("Minimum of the constraint polynomial is %g --> finished successfully!", minvalue)
      self.__results = OptimizeResult(
        x=x / self.__multiplycoeffs,
        success=True,
        status=1,
        nit=len(self.__cuttingplanes)+1,
        maxcv=0,
        message="finished successfully",
        fun=prob.value - self.__funatminimum
      )
      return

    constantindex = self.constantindex(minimizepolynomial)

    if (
      len(self.__cuttingplanes)+1 >= self.__maxiter
      #and constantindex is None
    ):
      self.__logger.info("Minimum of the constraint polynomial is %g\nReached the max number of iterations %d", minvalue, self.__maxiter)
      self.__results = OptimizeResult(
        x=x / self.__multiplycoeffs,
        success=True,
        status=4,
        nit=len(self.__cuttingplanes)+1,
        maxcv=-minvalue,
        message="reached max number of iterations"
                #", no constant term to increase!"
                ,
        fun=self.__tominimize.value - self.__funatminimum
      )
      return

    if (
      constantindex is not None and -minvalue < x[constantindex] * self.__maxfractionaladjustment
      or len(self.__cuttingplanes)+1 >= self.__maxiter
    ):
      self.__logger.info("Minimum of the constraint polynomial is %g", minvalue)
      if -minvalue > x[constantindex] * self.__maxfractionaladjustment:
        self.__logger.info("Reached the max number of iterations %d", self.__maxiter)
        extramessage = "reached max number of iterations.  "
        status = 3
      else:
        extramessage = ""
        status = 2

      oldconstant = x[constantindex]
      multiplier = 1
      evalconstraintkwargs = {}
      if hasattr(minimizepolynomial, "permutation"):
        evalconstraintkwargs["permutationdict"] = minimizepolynomial.permutation

      adjustiter = 0
      while minvalue < 0:
        assert adjustiter < 200
        adjustiter += 1
        lastconstant = x[constantindex]
        lastminimum = minvalue
        #print minimizepolynomial
        #print x
        #print minimizepolynomial.get("permutation", None), constantindex, x[constantindex], minvalue
        #print
        #raw_input()
        x[constantindex] -= minvalue - multiplier*np.finfo(float).eps
        minimizepolynomial = self.evalconstraint(x, **evalconstraintkwargs)
        minvalue = minimizepolynomial.fun
        if x[constantindex] == lastconstant or lastminimum == minvalue: multiplier += 1

      if x[constantindex] / oldconstant - 1 < self.__maxfractionaladjustment:
        self.__logger.info("Multiply constant term by (1+%g) --> new minimum of the constraint polynomial is %g", x[constantindex] / oldconstant - 1, minvalue)
        self.__logger.info("Approximate minimum of the target function is {} at {}".format(self.__tominimize.value - self.__funatminimum, x))
        self.__results = OptimizeResult(
          x=x / self.__multiplycoeffs,
          success=True,
          status=status,
          nit=len(self.__cuttingplanes)+1,
          maxcv=0,
          message=extramessage + "multiplied constant term by (1+{}) to get within constraint".format(x[constantindex] / oldconstant - 1),
          fun=self.__tominimize.value - self.__funatminimum
        )
        return

    self.__logger.info("Minimum of the constraint polynomial is {} at {} --> adding a new constraint using this minimum:\n{}".format(minvalue, minimizepolynomial.x, minimizepolynomial.linearconstraint))
    self.__cuttingplanes.append(
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
  polynomialclass = staticmethod(PolynomialNd)
  def polynomialargs(self, coeffs=None):
    if coeffs is None: coeffs = [1]*self.xsize
    return 2, 1, coeffs
  evalconstraint = staticmethod(minimizequadratic)
  def constantindex(self, minimizepolynomialresult): return 0

class CuttingPlaneMethod1DQuartic(CuttingPlaneMethodBase):
  xsize = 5
  polynomialclass = staticmethod(PolynomialNd)
  def polynomialargs(self, coeffs=None):
    if coeffs is None: coeffs = [1]*self.xsize
    return 4, 1, coeffs
  evalconstraint = staticmethod(minimizequartic)
  def constantindex(self, minimizepolynomialresult): return 0

class CuttingPlaneMethodMultiDimensional(CuttingPlaneMethodBase):
  def __init__(self, *args, **kwargs):
    self.__usepermutations = kwargs.pop("usepermutations", "asneeded")
    super(CuttingPlaneMethodMultiDimensional, self).__init__(*args, **kwargs)

  def minimizepolynomialfunction(self, coeffs, **kwargs):
    polynomial = self.polynomialclass(*self.polynomialargs(coeffs))
    if "permutationdict" in kwargs:
      function = polynomial.minimize_permutation
    else:
      function = {
        True: polynomial.minimize_permutations,
        False: polynomial.minimize,
        "asneeded": polynomial.minimize_permutationsasneeded,
      }[self.__usepermutations]
    return function(**kwargs)

  def constantindex(self, minimizepolynomialresult):
    permutation = minimizepolynomialresult.get("permutation", {v: v for v in self.polynomialvariables})

    constantvariables = next(iter(self.polynomialclass(*self.polynomialargs()).dehomogenizevariablesorder))
    permutedtoconstant = {permutation[v] for v in constantvariables}

    #for k, v in permutation.iteritems():
    #  if v == "1": permutedtoconstant = k
    for i, monomial in enumerate(self.monomials):
      if set(monomial.keys()) <= permutedtoconstant:
        return i
    assert False

class CuttingPlaneMethodPolynomialNd(CuttingPlaneMethodMultiDimensional):
  polynomialclass = HomogeneousPolynomialNd
  def polynomialargs(self, coeffs=None):
    if coeffs is None: coeffs = [1]*self.xsize
    return self.degree, self.nvariables+1, coeffs

  @abc.abstractproperty
  def degree(self): "can be a class member"
  @abc.abstractproperty
  def nvariables(self): "can be a class member"

class CuttingPlaneMethodPolynomialNdSimple(CuttingPlaneMethodPolynomialNd):
  def evalconstraint(self, coeffs, **kwargs):
    return self.minimizepolynomialfunction(coeffs, **kwargs)

  @property
  def xsize(self): return special.comb(self.nvariables+1, self.degree, repetition=True, exact=True)

class CuttingPlaneMethod_InsertZeroAtIndices(CuttingPlaneMethodMultiDimensional):
  def evalconstraint(self, coeffs, **kwargs):
    coeffs = iter(coeffs)
    newcoeffs = np.array([0 if i in self.insertzeroatindices else next(coeffs) for i in xrange(self.xsize)])
    for remaining in coeffs: assert False
    return self.minimizepolynomialfunction(newcoeffs, **kwargs)

  def constantindex(self, minimizepolynomialresult):
    permutation = minimizepolynomialresult.get("permutation", {"1": "1"})
    permutedtoconstant = permutation["1"]
    for i, monomial in enumerate(self.monomials):
      if set(monomial.keys()) == {permutedtoconstant}:
        return i
    assert permutedtoconstant in self.variableswithnoquarticterm
    return None

  @abc.abstractproperty
  def variableswithnoquarticterm(self): "can be a class member"
  @abc.abstractproperty
  def insertzeroatindices(self): "can be a class member"

  def monomials(self, coeffs=None):
    if coeffs is None: coeffs = [1] * self.xsize
    coeffs = iter(coeffs)
    newcoeffs = np.array([0 if i in self.insertzeroatindices else next(coeffs) for i in xrange(self.xsize)])
    for remaining in coeffs: assert False
    result = super(CuttingPlaneMethod_InsertZeroAtIndices, self).monomials(coeffs=newcoeffs)
    for _ in sorted(self.insertzeroatindices, reverse=True):
      del result[_]
    return result
  @property
  def useconstraintindices(self):
    result = range(len(super(CuttingPlaneMethod_InsertZeroAtIndices, self).monomials))
    for _ in sorted(self.insertzeroatindices, reverse=True):
      del result[_]
    return result
  @property
  def xsize(self):
    result = super(CuttingPlaneMethod_InsertZeroAtIndices, self).xsize - len(self.insertzeroatindices)
    if result != self.expectedxsize is not None:
      raise ValueError("Expected xsize to be {} but it's {}".format(self.expectedxsize, result))
    return result
  expectedxsize = None

class CuttingPlaneMethod3DQuadratic(CuttingPlaneMethodPolynomialNdSimple):
  degree = 2
  nvariables = 3

class CuttingPlaneMethod4DQuadratic(CuttingPlaneMethodPolynomialNdSimple):
  degree = 2
  nvariables = 4

class CuttingPlaneMethod4DQuartic(CuttingPlaneMethodPolynomialNdSimple):
  degree = 4
  nvariables = 4

class CuttingPlaneMethod4DQuartic_4thVariableQuadratic(CuttingPlaneMethod_InsertZeroAtIndices, CuttingPlaneMethod4DQuartic):
  expectedxsize = 65
  def insertzeroatindices():
    for idx, variables in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs):
      if variables["z"] >= 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())
  variableswithnoquarticterm = "z",

class CuttingPlaneMethod4DQuartic_4thVariableNoCubic(CuttingPlaneMethod_InsertZeroAtIndices, CuttingPlaneMethod4DQuartic):
  expectedxsize = 66
  def insertzeroatindices():
    for idx, variables in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs):
      if variables["z"] == 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())
  variableswithnoquarticterm = ()

class CuttingPlaneMethod4DQuartic_1stVariableOnlyEven(CuttingPlaneMethod_InsertZeroAtIndices, CuttingPlaneMethod4DQuartic):
  expectedxsize = 46
  def insertzeroatindices():
    for idx, variables in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs):
      if variables["w"] in (1, 3):
        yield idx
  insertzeroatindices = list(insertzeroatindices())
  variableswithnoquarticterm = ()

class CuttingPlaneMethod4DQuartic_4thVariableQuadratic_1stVariableOnlyEven(CuttingPlaneMethod_InsertZeroAtIndices, CuttingPlaneMethod4DQuartic):
  expectedxsize = 42
  def insertzeroatindices():
    for idx, variables in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs):
      if variables["w"] in (1, 3) or variables["z"] >= 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())
  variableswithnoquarticterm = "z",

class CuttingPlaneMethod4DQuartic_4thVariableNoCubic_1stVariableOnlyEven(CuttingPlaneMethod_InsertZeroAtIndices, CuttingPlaneMethod4DQuartic):
  expectedxsize = 43
  def insertzeroatindices():
    for idx, variables in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs):
      if variables["w"] in (1, 3) or variables["z"] == 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())
  variableswithnoquarticterm = ()

class CuttingPlaneMethodDoubleQuadratic(CuttingPlaneMethodMultiDimensional):
  polynomialclass = HomogeneousDoubleQuadratic
  def polynomialargs(self, coeffs=None):
    if coeffs is None: coeffs = [1]*self.xsize
    return self.nvariables1, self.nvariables2+1, coeffs

  @abc.abstractproperty
  def nvariables1(self): "can be a class member"
  @abc.abstractproperty
  def nvariables2(self): "can be a class member"

class CuttingPlaneMethodDoubleQuadraticSimple(CuttingPlaneMethodDoubleQuadratic):
  def evalconstraint(self, coeffs, **kwargs):
    return self.minimizepolynomialfunction(coeffs, **kwargs)

  def polynomialargs(self, coeffs=None):
    if coeffs is None: coeffs = [1]*self.xsize
    return self.nvariables1+1, self.nvariables2+1, coeffs

  @property
  def xsize(self): return special.comb(self.nvariables1+1, 2, repetition=True, exact=True) * special.comb(self.nvariables2+1, 2, repetition=True, exact=True)

class CuttingPlaneMethod1DQuadraticBy4DQuadratic(CuttingPlaneMethodDoubleQuadraticSimple):
  nvariables1 = 1
  nvariables2 = 4

def cuttingplanemethod1dquadratic(*args, **kwargs):
  return CuttingPlaneMethod1DQuadratic(*args, **kwargs).run()
def cuttingplanemethod1dquartic(*args, **kwargs):
  return CuttingPlaneMethod1DQuartic(*args, **kwargs).run()
def cuttingplanemethod3dquadratic(*args, **kwargs):
  return CuttingPlaneMethod3DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquartic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablequadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablenocubic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableNoCubic(*args, **kwargs).run()
def cuttingplanemethod4dquartic_1stvariableonlyeven(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_1stVariableOnlyEven(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablequadratic_1stvariableonlyeven(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableQuadratic_1stVariableOnlyEven(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablenocubic_1stvariableonlyeven(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableNoCubic_1stVariableOnlyEven(*args, **kwargs).run()

class CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_Step2(CuttingPlaneMethodMultiDimensional):
  z34indices = [i for i, monomial in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs) if monomial["z"] >= 3]
  xsize = 5
  def constantindex(self, minimizepolynomialresult): return None
  assert len(z34indices) == xsize

  useconstraintindices = range(70)
  monomials = list(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs)
  unusedmonomials = []
  for _ in sorted(range(70), reverse=True):
    if _ in z34indices: continue
    assert useconstraintindices[_] == _
    del useconstraintindices[_]
    unusedmonomials.insert(0, monomials.pop(_))
  del _

  def evalconstraint(self, coeffs, othercoeffs):
    coeffs = iter(coeffs)
    othercoeffs = iter(othercoeffs)
    newcoeffs = np.array([next(coeffs) if i in self.z34indices else next(othercoeffs) for i in xrange(70)])
    for remaining in coeffs: assert False
    for remaining in othercoeffs: assert False
    return self.minimizepolynomialfunction(4, 4, newcoeffs)

  @property
  def maxpowerindices(self):
    maxpowers = {
      varname: max(monomial[varname] for monomial in self.monomials)
        for varname in self.polynomialvariables
        if varname == "z"
    }
    for v, p in maxpowers.iteritems():
      if p%2: raise ValueError("max power of {} is odd: {}".format(v, p))
    result = []
    for i, monomial in enumerate(self.monomials):
      if any(monomial[v] == p for v, p in maxpowers.iteritems()):
        result.append(i)
    return result

  @classmethod
  def modifymoreargs(cls):
    def modifyothercoeffs(multiplyvariables, othercoeffs):
      multiplyothercoeffs = 1/np.array([np.prod([multiplyvariables[variable] for variable in monomial.elements()]) for monomial in cls.unusedmonomials])
      return multiplyothercoeffs * othercoeffs
    return modifyothercoeffs,

class CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_1stVariableOnlyEven_Step2(CuttingPlaneMethodMultiDimensional):
  z34indices = [i for i, monomial in enumerate(m for m in HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs if m["w"]%2==0) if monomial["z"] >= 3]
  xsize = 4
  def constantindex(self, minimizepolynomialresult): return None
  assert len(z34indices) == xsize

  useconstraintindices = range(46)
  monomials = list(m for m in HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs if m["w"]%2==0)
  unusedmonomials = []
  for _ in sorted(range(46), reverse=True):
    if _ in z34indices: continue
    assert useconstraintindices[_] == _
    del useconstraintindices[_]
    unusedmonomials.insert(0, monomials.pop(_))
  del _

  def evalconstraint(self, coeffs, othercoeffs):
    coeffs = iter(coeffs)
    othercoeffs = iter(othercoeffs)
    newcoeffs = np.array([next(coeffs) if i in self.z34indices else next(othercoeffs) for i in xrange(46)])
    for remaining in coeffs: assert False
    for remaining in othercoeffs: assert False
    return self.minimizepolynomialfunction(4, 4, newcoeffs)

  @property
  def maxpowerindices(self):
    maxpowers = {
      varname: max(monomial[varname] for monomial in self.monomials)
        for varname in self.polynomialvariables
        if varname == "z"
    }
    for v, p in maxpowers.iteritems():
      if p%2: raise ValueError("max power of {} is odd: {}".format(v, p))
    result = []
    for i, monomial in enumerate(self.monomials):
      if any(monomial[v] == p for v, p in maxpowers.iteritems()):
        result.append(i)
    return result

  @classmethod
  def modifymoreargs(cls):
    def modifyothercoeffs(multiplyvariables, othercoeffs):
      multiplyothercoeffs = 1/np.array([np.prod([multiplyvariables[variable] for variable in monomial.elements()]) for monomial in cls.unusedmonomials])
      return multiplyothercoeffs * othercoeffs
    return modifyothercoeffs,

def cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs) if monomial["z"] >= 3]

  assert np.all(x0[z34indices] == 0)

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])

  result = cuttingplanemethod4dquartic_4thvariablequadratic(x0withoutz34, sigmawithoutz34, *args, **kwargs)

  x = iter(result.x)
  result.x = np.array([0 if i in z34indices else next(x) for i in xrange(len(x0))])
  for remaining in x: assert False

  result.message += " (4th variable is only quadratic)"

  return result

def cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic_1stvariableonlyeven(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(m for m in HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs if m["w"]%2==0) if monomial["z"] >= 3]

  assert np.all(x0[z34indices] == 0)

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])

  result = cuttingplanemethod4dquartic_4thvariablequadratic_1stvariableonlyeven(x0withoutz34, sigmawithoutz34, *args, **kwargs)

  x = iter(result.x)
  result.x = np.array([0 if i in z34indices else next(x) for i in xrange(len(x0))])
  for remaining in x: assert False

  result.message += " (4th variable is only quadratic and 1st variable is only even)"

  return result

def cuttingplanemethod4dquartic_4thvariablezerocubic(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs) if monomial["z"] == 3]

  assert np.all(x0[z34indices] == 0)

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])

  result = cuttingplanemethod4dquartic_4thvariablenocubic(x0withoutz34, sigmawithoutz34, *args, **kwargs)

  x = iter(result.x)
  result.x = np.array([0 if i in z34indices else next(x) for i in xrange(len(x0))])
  for remaining in x: assert False

  result.message += " (4th variable has no cubic terms)"

  return result

def cuttingplanemethod4dquartic_4thvariablezerocubic_1stvariableonlyeven(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(m for m in HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs if m["w"]%2==0) and monomial["z"] == 3]

  assert np.all(x0[z34indices] == 0)

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])

  result = cuttingplanemethod4dquartic_4thvariablenocubic_1stvariableonlyeven(x0withoutz34, sigmawithoutz34, *args, **kwargs)

  x = iter(result.x)
  result.x = np.array([0 if i in z34indices else next(x) for i in xrange(len(x0))])
  for remaining in x: assert False

  result.message += " (4th variable has no cubic terms and 1st variable is only even)"

  return result

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs) if monomial["z"] >= 3]

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])
  x0z34 = np.array([_ for i, _ in enumerate(x0) if i in z34indices])
  sigmaz34 = np.array([_ for i, _ in enumerate(sigma) if i in z34indices])

  result1 = cuttingplanemethod4dquartic_4thvariablequadratic(x0withoutz34, sigmawithoutz34, *args, **kwargs)
  try:
    if "maxiter" in kwargs:
      kwargs["maxiter"] /= 10
    result2 = cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_step2(x0z34, sigmaz34, *args, moreargsforevalconstraint=(result1.x,), **kwargs)
  except Exception as e:
    result = OptimizeResult({
      k+"_step1": v for k, v in result1.iteritems()
    })
    result.update({
      k: v for k, v in result1.iteritems()
    })
    result.update({
      "error_step2": e,
      "status": 5,  #overrides the one from result1.iteritems
    })
    x1 = iter(result1.x)
    result.x = np.array([0 if i in z34indices else next(x1) for i in xrange(len(x0))])
    for remaining in x1: assert False
    return result

  result = OptimizeResult({
    k+"_step1": v for k, v in result1.iteritems()
  })
  result.update({
    k+"_step2": v for k, v in result2.iteritems()
  })
  x1 = iter(result1.x)
  x2 = iter(result2.x)
  result.x = np.array([next(x2) if i in z34indices else next(x1) for i in xrange(len(x0))])
  for remaining in x1: assert False
  for remaining in x2: assert False

  result.message = "4th variable is small beyond quadratic, 2 steps.\nFirst: {}\nSecond: {}".format(result1.message, result2.message)
  result.nit = result1.nit + result2.nit
  result.maxcv = result1.maxcv + result2.maxcv
  result.fun = result1.fun + result2.fun
  result.status = max(result1.status, result2.status)

  return result

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_step2(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_Step2(*args, **kwargs).run()

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_1stvariableonlyeven(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(m for m in HomogeneousPolynomialNd(4, 5, [1]*70).monomialswithoutcoeffs if m["w"]%2==0) if monomial["z"] >= 3]

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])
  x0z34 = np.array([_ for i, _ in enumerate(x0) if i in z34indices])
  sigmaz34 = np.array([_ for i, _ in enumerate(sigma) if i in z34indices])

  result1 = cuttingplanemethod4dquartic_4thvariablequadratic_1stvariableonlyeven(x0withoutz34, sigmawithoutz34, *args, **kwargs)
  try:
    if "maxiter" in kwargs:
      kwargs["maxiter"] /= 10
    result2 = cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_1stvariableonlyeven_step2(x0z34, sigmaz34, *args, moreargsforevalconstraint=(result1.x,), **kwargs)
  except Exception as e:
    result = OptimizeResult({
      k+"_step1": v for k, v in result1.iteritems()
    })
    result.update({
      k: v for k, v in result1.iteritems()
    })
    result.update({
      "error_step2": e,
      "status": 5,  #overrides the one from result1.iteritems
    })
    x1 = iter(result1.x)
    result.x = np.array([0 if i in z34indices else next(x1) for i in xrange(len(x0))])
    for remaining in x1: assert False
    return result

  result = OptimizeResult({
    k+"_step1": v for k, v in result1.iteritems()
  })
  result.update({
    k+"_step2": v for k, v in result2.iteritems()
  })
  x1 = iter(result1.x)
  x2 = iter(result2.x)
  result.x = np.array([next(x2) if i in z34indices else next(x1) for i in xrange(len(x0))])
  for remaining in x1: assert False
  for remaining in x2: assert False

  result.message = "4th variable is small beyond quadratic, 2 steps.\nFirst: {}\nSecond: {}".format(result1.message, result2.message)
  result.nit = result1.nit + result2.nit
  result.maxcv = result1.maxcv + result2.maxcv
  result.fun = result1.fun + result2.fun
  result.status = max(result1.status, result2.status)

  return result

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_1stvariableonlyeven_step2(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_1stVariableOnlyEven_Step2(*args, **kwargs).run()

if __name__ == "__main__":
  a = np.array([[1, 2.]]*45)
  a[3,:] *= -5
  print CuttingPlaneMethod1DQuadraticBy4DQuadratic(
    a,
    abs(a),
    maxfractionaladjustment=1e-6,
    printaswego=True,
  ).run()
