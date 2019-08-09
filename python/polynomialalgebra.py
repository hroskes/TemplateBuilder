#!/usr/bin/env python

"""
All functions in this module take the coefficients in this order:
1, x, y, z, ..., x^2, xy, xz, ..., x^3, x^2y, x^2z, ......

linearformula, quadraticformula, and cubicformula solve a + bx (+ cx^2 (+dx^3)) = 0
and return the results as a numpy array of solutions.

The minimize functions all return an OptimizeResults.
If the minimum is actually -inf, the success will be False and the reported results
will be a number less than -1e6, with some set of xs that produce that result.
"""

from __future__ import division

import abc, collections, functools, itertools, subprocess

import numpy as np

import hom4pswrapper
from moremath import closebutnotequal, notnan
from optimizeresult import OptimizeResult

@notnan
def linearformula(coeffs):
  """
  solve a + bx = 0
  """
  a, b = coeffs
  if b==0: raise ValueError("function is a constant")
  return np.array([-a/b])

@notnan
def quadraticformula(coeffs):
  """
  solve a + bx + cx^2 = 0

  NOTE this is consistent with the convention here, but the opposite of the normal convention
  """
  a, b, c = coeffs
  if c == 0: return linearformula(a, b)
  return (-b + np.array([1, -1]) * (b**2 - 4*c*a + 0j)**0.5) / (2*c)

@notnan
def cubicformula(coeffs):
  """
  solve a + bx + cx^2 + dx^3 = 0

  NOTE this is consistent with the convention here, but the opposite of the normal convention (e.g. wikipedia)
  """
  a, b, c, d = coeffs

  if d==0: return quadraticformula(a, b, c)

  Delta0 = c**2 - 3*d*b
  Delta1 = 2*c**3 - 9*d*c*b + 27*d**2*a

  if Delta0 == Delta1 == 0: return np.array([-c / (3*d)])

  C = ((Delta1 + (1 if Delta1>0 else -1) * (Delta1**2 - 4*Delta0**3 + 0j)**0.5) / 2) ** (1./3)

  xi = 0.5 * (-1 + (3**0.5)*1j)

  return np.array([-1/(3.*d) * (c + xi**k*C + Delta0/(xi**k * C)) for k in range(3)])

def minimizeconstant(coeffs):
  """
  minimize y=a
  """
  a, = coeffs
  return OptimizeResult(
    x=np.array([0]),
    success=True,
    status=2,
    message="function is constant",
    fun=a,
    linearconstraint=np.array([1]),
  )

def minimizelinear(coeffs):
  """
  minimize y=a+bx
  """
  a, b = coeffs
  if not b:
    result = minimizeconstant(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([0, x])
    return result
  x = linearformula((a+2e6, b))[0]
  fun = a + b*x
  assert fun < -1e6
  return OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is linear, no minimum",
    fun=fun,
    linearconstraint=np.array([0, x]),
  )

def minimizequadratic(coeffs):
  """
  minimize y=a+bx+c
  """
  a, b, c = coeffs
  if c == 0:
    result = minimizelinear(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([0, 0, 1])
    return result
  if c < 0:
    x = quadraticformula((a+max(2e6, -a+2e6), b, c))[0]
    assert np.imag(x) == 0, x
    x = np.real(x)
    fun = a + b*x + c*x**2
    assert fun < -1e6
    return OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quadratic, no minimum",
      fun=fun,
      linearconstraint=np.array([0, 0, 1]),
    )

  x = linearformula([b, 2*c])[0]
  fun = a + b*x + c*x**2

  return OptimizeResult(
    x=np.array([x]),
    success=True,
    status=1,
    message="function is quadratic",
    fun=fun,
    linearconstraint=np.array([1, x, x**2]),
  )

def minimizecubic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3
  """
  a, b, c, d = coeffs
  if d == 0:
    result = minimizequadratic(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.concatenate((result.linearconstraint, [0]))
    return result
  x = [_ for _ in cubicformula((a+2e6, b, c, d)) if abs(np.imag(_)) < 1e-12][0]
  x = np.real(x)
  fun = a + b*x + c*x**2 + d*x**3
  assert fun < -1e6
  return OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is cubic, no minimum",
    fun=fun,
    linearconstraint=np.array([0, 0, 0, x**3])
  )

def minimizequartic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3+ex^4
  """
  a, b, c, d, e = coeffs
  if e == 0:
    result = minimizecubic(coeffs[:-1])
    x = result.x[0]
    if result.linearconstraint[-1]:
      result.linearconstraint = np.array([0, 0, 0, 0, 1])
    else:
      result.linearconstraint = np.concatenate((result.linearconstraint, [0]))
    return result
  if e < 0:
    x = 1
    fun = 0
    while fun > -1e6:
      x *= 10
      fun = a + b*x + c*x**2 + d*x**3 + e*x**4
    return OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quartic, no minimum",
      fun=fun,
      linearconstraint = np.array([0, 0, 0, 0, 1]),
    )

  flatpoints = cubicformula(np.array([b, 2*c, 3*d, 4*e]))
  minimum = float("inf")
  x = None
  for flatpoint in flatpoints:
    if abs(np.imag(flatpoint)) > 1e-12: continue
    flatpoint = np.real(flatpoint)
    newminimum = a + b*flatpoint + c*flatpoint**2 + d*flatpoint**3 + e*flatpoint**4
    if newminimum < minimum:
      minimum = newminimum
      x = flatpoint
  if np.isnan(minimum) or x is None: assert False, (coeffs, x, minimum)

  return OptimizeResult(
    x=np.array([x]),
    success=True,
    status=1,
    message="function is quartic",
    fun=np.real(minimum),
    linearconstraint = np.array([1, x, x**2, x**3, x**4]),
  )

def getnvariableletters(n, frombeginning=False):
  abc = "abcdefghijklmnopqrstuvwxyz"
  if frombeginning:
    return abc[:n]
  else:
    return abc[-n:]

class VariablePowers(collections.Counter):
  def __hash__(self):
    return hash(frozenset(i for i in self.items() if i != "1"))
  def __eq__(self, other):
    return set(i for i in self.items() if i != "1") == set(i for i in other.items() if i != "1")
  def __ne__(self, other):
    return not self == other

  @property
  def degree(self): return sum(1 for variable in self.elements() if variable != "1")
  @property
  def maxvariablepower(self):
    return max(itertools.chain([0], (power for variable, power in self.items() if variable != "1")))
  def __call__(self, **variablevalues):
    variablevalues["1"] = 1
    return np.prod([variablevalues[variable] for variable in self.elements()])

  def __str__(self):
    if not any(self.values()): return "1"
    return "*".join(self.elements())

  def homogenize(self, withvariable, todegree):
    if self.degree == todegree or not self: return self
    assert self.degree < todegree
    newctr = VariablePowers(self)
    if withvariable != "1": assert withvariable not in newctr
    newctr[withvariable] = todegree - self.degree
    return newctr
  def dehomogenize(self, withvariables):
    newctr = VariablePowers(self)
    for withvariable in withvariables:
      newctr[withvariable] = 0
    if not any(newctr.values()): newctr["1"] = 1
    return VariablePowers(newctr)
  def permutevariables(self, permutationdict):
    return VariablePowers(permutationdict[variable] for variable in self.elements())

class Monomial(collections.namedtuple("Monomial", "coeff variablepowers")):
  def __new__(cls, coeff, variablepowers):
    variablepowers = VariablePowers(variablepowers)

    return super(Monomial, cls).__new__(cls, coeff, variablepowers)

  @property
  def degree(self):
    return self.variablepowers.degree
  @property
  def maxvariablepower(self):
    return self.variablepowers.maxvariablepower
  def __nonzero__(self):
    if isinstance(self.coeff, np.ndarray):
      return bool(np.any(self.coeff))
    else:
      return bool(self.coeff)

  def __call__(self, **variablevalues):
    return self.coeff * self.variablepowers(**variablevalues)
  def derivative(self, variable):
    newctr = collections.Counter(self.variablepowers)
    newcoeff = self.coeff * newctr[variable]
    newctr[variable] -= 1
    return Monomial(newcoeff, newctr)
  def __str__(self):
    return "{!r}*{}".format(self.coeff, self.variablepowers)

  def permutevariables(self, permutationdict):
    return Monomial(self.coeff, self.variablepowers.permutevariables(permutationdict))
  def homogenize(self, withvariable, todegree):
    return Monomial(self.coeff, self.variablepowers.homogenize(withvariable, todegree))
  def dehomogenize(self, withvariables):
    return Monomial(self.coeff, self.variablepowers.dehomogenize(withvariables))

class PolynomialBase(object):
  def __init__(self):
    self.__nmonomials = sum(1 for m in self.monomials)
    self.__degree = max(monomial.degree for monomial in self.monomials)
    self.__maxvariablepower = max(monomial.maxvariablepower for monomial in self.monomials)
    allvariables = set(itertools.chain(*(monomial.variablepowers.elements() for monomial in self.monomials)))
    allvariables.discard("1")

    self.__nvariables = len(allvariables)
    for letter in allvariables:
      if letter not in self.variableletters: raise ValueError("monomial letter {!r} is invalid, should be one of {}".format(letter, self.variableletters))

    seen = set()
    for monomial in self.monomials:
      if monomial.variablepowers in seen:
        raise ValueError("Got a duplicate monomial: {}".format(variablepowers))
      seen.add(monomial.variablepowers)

  @abc.abstractproperty
  def monomials(self): "list of Monomial objects"

  @property
  def nmonomials(self): return self.__nmonomials
  @property
  def degree(self): return self.__degree
  @property
  def maxvariablepower(self): return self.__maxvariablepower

  @property
  def ishomogeneous(self): return all(m.degree == self.degree for m in self.monomials if m)

  @property
  def nvariables(self): return self.__nvariables

  @abc.abstractproperty
  def variableletters(self): pass
  @abc.abstractproperty
  def dehomogenizevariablesorder(self): pass

  def __call__(self, variablearray=None, **kwargs):
    if variablearray is not None:
      if kwargs:
        raise TypeError("Can't provide both variablearray and kwargs")
      assert len(variablearray) == self.nvariables
      kwargs = {letter: x for letter, x in itertools.izip_longest(self.variableletters, variablearray)}
    return sum(monomial(**kwargs) for monomial in self.monomials)

  @property
  def boundarypolynomial(self):
    monomials = []
    sawmaxpower = {letter: False for letter in self.variableletters}
    for monomial in self.monomials:
      if monomial.degree < self.degree: continue
      monomials.append(monomial)
      if monomial:
        for letter, power in monomial.variablepowers.iteritems():
          if power == self.maxvariablepower:
            sawmaxpower[letter] = True

    result = ExplicitPolynomial(monomials, self.variableletters, self.dehomogenizevariablesorder).dehomogenize()

    for letter in self.variableletters:
      if not sawmaxpower[letter]:
        raise DegeneratePolynomialError(self, letter, self.maxvariablepower)

    return result

  def derivative(self, variable):
    return ExplicitPolynomial((monomial.derivative(variable) for monomial in self.monomials), self.variableletters, None)

  def permutevariables(self, permutationdict):
    if not self.ishomogeneous: raise ValueError("Can't permute a non-homogeneous polynomial")
    return ExplicitPolynomial((monomial.permutevariables(permutationdict) for monomial in self.monomials), self.variableletters, self.dehomogenizevariablesorder)

  @property
  def gradient(self):
    return [self.derivative(variable) for variable in self.variableletters]

  def __str__(self):
    return " + ".join(str(m) for m in self.monomials if m)

  def homogenize(self, withvariable):
    return ExplicitPolynomial((monomial.homogenize(withvariable, self.degree) for monomial in self.monomials), self.variableletters + (withvariable,), None)
  def dehomogenize(self):
    if not self.ishomogeneous: raise ValueError("Can't dehomogenize\n{}\nwhich isn't homogeneous".format(self))
    if self.dehomogenizevariablesorder is None: raise ValueError("Can't dehomogenize\n{}\nwhich doesn't have a dehomogenizevariablesorder")
    order = iter(self.dehomogenizevariablesorder)
    dehomogenizewith = next(order)
    return ExplicitPolynomial((monomial.dehomogenize(dehomogenizewith) for monomial in self.monomials), [_ for _ in self.variableletters if _ not in dehomogenizewith], tuple(order))

  def setsmallestcoefficientsto0(self):
    coeffs = np.array([m.coeff for m in self.monomials])
    newcoeffs = []
    setto0 = []
    newmonomials = []
    biggest = max(abs(coeffs))
    smallest = min(abs(coeffs[np.nonzero(coeffs)]))
    for monomial in self.monomials:
      coeff = monomial.coeff
      if coeff == 0 or np.log(biggest / abs(coeff)) < np.log(abs(coeff) / smallest):
        newcoeffs.append(coeff)
        newmonomials.append(monomial)
      else:
        setto0.append(coeff)
        newcoeffs.append(0)
        newmonomials.append(Monomial(0, monomial.variablepowers))
    newcoeffs = np.array(newcoeffs)
    setto0 = np.array(setto0)
    if np.log10(min(abs(newcoeffs[np.nonzero(newcoeffs)])) / max(abs(setto0))) < np.log10(biggest / smallest) / 3:
      #no big gap between the big ones and the small ones
      raise NoCoeffGapError
    return ExplicitPolynomial(newmonomials, self.variableletters, self.dehomogenizevariablesorder)

  def findcriticalpoints(self, verbose=False, cmdlinestotry=("smallparalleltdeg",), homogenizecoeffs=None, boundarycriticalpoints=[], setsmallestcoefficientsto0=False):
    if self.ishomogeneous:
      raise ValueError("You need to dehomogenize the polynomial before you find critical points")

    if self.degree == 2:
      variableletters = self.variableletters
      #Ax=b
      A = np.zeros((n, n))
      b = np.zeros((n, 1))
      for derivative, row, constant in itertools.izip_longest(self.gradient, A, b):
        for coeff, xs in derivative:
          xs = list(xs.elements())
          assert len(xs) <= 1
          if not xs or xs[0] == "1":
            constant[0] = -coeff  #note - sign here!  Because the gradient should be Ax + (-b)
          else:
            row[variableletters.index(xs[0])] = coeff
      return np.linalg.solve(A, b).T


    gradient = self.gradient
    extraequations = []
    variableletters = self.variableletters

    if homogenizecoeffs is not None:
      gradient = [derivative.homogenize("alpha") for derivative in gradient]
      linearpolynomial = ExplicitPolynomial(
        (
          Monomial(coeff, [variable]) for coeff, variable in itertools.izip_longest(
            homogenizecoeffs,
            itertools.chain(["alpha"], self.variableletters, ["1"])
          )
        ),
        self.variableletters+("alpha",),
        None,
      )
      extraequations = [str(linearpolynomial)+";"]

      variableletters = ("alpha",) + variableletters

    gradientstrings = [str(derivative)+";" for derivative in gradient]
    stdin = "\n".join(["{"] + gradientstrings + extraequations + ["}"])

    errors = []
    for cmdline in cmdlinestotry:
      try:
        result = hom4pswrapper.runhom4ps(stdin, whichcmdline=cmdline, verbose=verbose)
        assert result.variableorder == variableletters, (result.variableorder, variableletters)
      except hom4pswrapper.Hom4PSTimeoutError as e:
        pass
      except hom4pswrapper.Hom4PSFailedPathsError as e:
        errors.append(e)
      except hom4pswrapper.Hom4PSDuplicateSolutionsError as e:
        errors.append(e)
      except hom4pswrapper.Hom4PSDivergentPathsError as e:
        if homogenizecoeffs is None:
          for cp in boundarycriticalpoints:
            try:
              newhomogenizecoeffs = np.concatenate(([1, 1], 1/cp, [1]))
            except RuntimeWarning as runtimewarning:
              if "divide by zero encountered in true_divide" in runtimewarning:
                continue #can't use this cp
            try:
              homogenizedresult = self.findcriticalpoints(verbose=verbose, cmdlinestotry=cmdlinestotry, homogenizecoeffs=newhomogenizecoeffs, setsmallestcoefficientsto0=setsmallestcoefficientsto0)
            except NoCriticalPointsError:
              pass
            else:
              for solution in e.realsolutions:
                if not any(np.allclose(solution, newsolution) for newsolution in homogenizedresult):
                  break
              else: #all old solutions are still there after homogenizing
                return homogenizedresult
        errors.append(e)
      else:
        solutions = result.realsolutions
        if homogenizecoeffs is not None:
          solutions = [solution[1:] / solution[0] for solution in solutions]
        return solutions

    if errors:
      if verbose:
        print "seeing if those calls gave different solutions, in case between them we have them all covered"

      solutions = []
      allclosekwargs = {"rtol": 1e-3, "atol": 1e-08}
      for error in errors:
        thesesolutions = error.realsolutions
        if homogenizecoeffs is not None:
          thesesolutions = [solution[1:] / solution[0] for solution in thesesolutions]

        while any(closebutnotequal(first, second, **allclosekwargs) for first, second in itertools.combinations(thesesolutions, 2)):
          allclosekwargs["rtol"] /= 2
          allclosekwargs["atol"] /= 2

        for newsolution in thesesolutions:
          if not any(closebutnotequal(newsolution, oldsolution, **allclosekwargs) for oldsolution in solutions):
            solutions.append(newsolution)

      numberofpossiblesolutions = min(len(e.solutions) + e.nfailedpaths + e.ndivergentpaths for e in errors)

      if len(solutions) > numberofpossiblesolutions:
        raise NoCriticalPointsError(self, moremessage="found too many critical points in the union of the different configurations", solutions=solutions)

      if len(solutions) == numberofpossiblesolutions:
        if verbose: print "we do"
        if homogenizecoeffs is not None:
          solutions = [solution[1:] / solution[0] for solution in solutions]
        return solutions

      if setsmallestcoefficientsto0:
        try:
          newsolutions = self.setsmallestcoefficientsto0().findcriticalpoints(verbose=verbose, cmdlinestotry=cmdlinestotry, homogenizecoeffs=homogenizecoeffs, boundarycriticalpoints=boundarycriticalpoints)
          for oldsolution in solutions:
            if verbose: print "checking if old solution {} is still here".format(oldsolution)
            if not any(np.allclose(oldsolution, newsolution, **allclosekwargs) for newsolution in newsolutions):
              if verbose: print "it's not"
              break  #removing this coefficient messed up one of the old solutions, so we can't trust the new ones
            if verbose: print "it is"
          else:  #removing this coefficient didn't mess up the old solutions
            return newsolutions
        except NoCoeffGapError:
          if verbose: print "can't set the smallest coefficients to 0, there's not a clear separation between big and small:\nbig candidates:{} --> range = {} - {}\nsmall candidates: {} --> range = {} - {}\n\nmore info: {} {}".format(newcoeffs[np.nonzero(newcoeffs)], min(abs(newcoeffs[np.nonzero(newcoeffs)])), max(abs(newcoeffs)), setto0, min(abs(setto0)), max(abs(setto0)), np.log10(min(abs(newcoeffs[np.nonzero(newcoeffs)])) / max(abs(setto0))), np.log10(biggest / smallest))

    else:
      solutions=None

    raise NoCriticalPointsError(self, moremessage="there are failed and/or divergent paths, even after trying different configurations and saving mechanisms", solutions=solutions)

  def minimize(self, verbose=False, **kwargs):
    if self.ishomogeneous:
      return self.dehomogenize().minimize(verbose=verbose, **kwargs)

    if self.nvariables == 1 and self.degree <= 4:
      coeffs = [m.coeff for m in sorted(self.monomials, key=lambda x: x.degree)]
      if self.degree == 0: return minimizeconstant(coeffs)
      if self.degree == 1: return minimizelinear(coeffs)
      if self.degree == 2: return minimizequadratic(coeffs)
      if self.degree == 3: return minimizecubic(coeffs)
      if self.degree == 4: return minimizequartic(coeffs)

    if all(set(monomial.variablepowers) == {"1"} for monomial in self.monomials if monomial):
      return OptimizeResult(
        x=np.array([0]*n),
        success=True,
        status=2,
        message="polynomial is constant",
        fun=[monomial.coeff for monomial in self.monomials if monomial][0],
        linearconstraint=np.array([1 if monomial else 0 for monomial in self.monomials])
      )

    #check the behavior around the sphere at infinity
    boundarykwargs = kwargs.copy()
    if kwargs.get("homogenizecoeffs") is not None:
      boundarykwargs["homogenizecoeffs"] = kwargs["homogenizecoeffs"][1:]
    boundarypolynomial = self.boundarypolynomial

    boundaryresult = boundarypolynomial.minimize(verbose=verbose, **boundarykwargs)
    if boundaryresult.fun < 0:
      x = np.concatenate(([1], boundaryresult.x))
      multiply = 1
      while self(x*multiply) > -1e6:
        multiply *= 10
        if multiply > 1e30: assert False

      linearconstraint = []
      boundarylinearconstraint = iter(boundaryresult.linearconstraint)

      for monomial in self.monomials:
        if monomial.degree == self.degree:
          linearconstraint.append(next(boundarylinearconstraint))
        else:
          linearconstraint.append(0)

      for remaining in boundarylinearconstraint: assert False

      return OptimizeResult(
        x=x*multiply,
        success=False,
        status=3,
        message="function goes to -infinity somewhere around the sphere at infinity",
        fun=self(x*multiply),
        linearconstraint=np.array(linearconstraint),
        boundaryresult=boundaryresult,
      )

    assert "boundarycriticalpoints" not in kwargs
    if hasattr(boundaryresult, "criticalpoints"):
      kwargs["boundarycriticalpoints"] = boundaryresult.criticalpoints

    criticalpoints = list(self.findcriticalpoints(verbose=verbose, **kwargs))
    if not criticalpoints:
      raise NoCriticalPointsError(self, moremessage="system of polynomials doesn't have any critical points")

    criticalpoints.sort(key=self)
    if verbose:
      for cp in criticalpoints:
        print cp, self(cp)
    minimumx = criticalpoints[0]
    minimum = self(minimumx)

    linearconstraint = ExplicitPolynomial(
      (Monomial(
        np.array([1 if i==j else 0 for i in xrange(self.nmonomials)]),
        monomial.variablepowers,
      ) for j, monomial in enumerate(self.monomials)),
      self.variableletters,
      None,
    )(minimumx)
    if not np.isclose(np.dot(linearconstraint, [monomial.coeff for monomial in self.monomials]), minimum, rtol=2e-2):
      raise ValueError("{} != {}??".format(np.dot(linearconstraint, [monomial.coeff for monomial in self.monomials]), minimum))

    return OptimizeResult(
      x=np.array(minimumx),
      success=True,
      status=1,
      message="gradient is zero at {} real points".format(len(criticalpoints)),
      fun=minimum,
      linearconstraint=linearconstraint,
      boundaryresult=boundaryresult,
      criticalpoints=criticalpoints,
    )

  def indexofmonomial(self, variablepowers):
    for i, monomial in enumerate(self.monomials):
      if monomial.variablepowers == variablepowers:
        return i
    raise IndexError

  def minimize_permutation(self, permutationdict, **kwargs):
    permuted = self.permutevariables(permutationdict)

    result = permuted.minimize(**kwargs)

    reverse = {v: k for k, v in permutationdict.iteritems()}

    if all(k == v for k, v in permutationdict.iteritems()): return result

    if (
      np.sign(np.dot(result.linearconstraint, [monomial.coeff for monomial in self.monomials])) != np.sign(result.fun)
      and np.sign(np.dot(result.linearconstraint, [monomial.coeff for monomial in self.monomials])) != 0
      and np.sign(result.fun) != 0
      and not np.isclose(np.dot(result.linearconstraint, [monomial.coeff for monomial in self.monomials]), result.fun) #numerical issues occasionally give +epsilon and -epsilon when you add in different orders
    ):
      raise ValueError("sign({}) != sign({})??".format(np.dot(result.linearconstraint, [monomial.coeff for monomial in self.monomials]), result.fun))

    return OptimizeResult(
      permutation=permutationdict,
      permutedresult=result,
      fun=result.fun,
      linearconstraint=result.linearconstraint,
      x=(result.x, "permuted")
    )

  def minimize_permutations(self, debugprint=False, permutationmode="best", **kwargs):
    xand1 = self.variableletters
    best = None
    signs = {1: [], -1: [], 0: []}
    for permutation in permutations_differentonesfirst(xand1):
      permutationdict = {orig: new for orig, new in itertools.izip(xand1, permutation)}
      try:
        result = self.minimize_permutation(permutationdict=permutationdict, **kwargs)
      except (NoCriticalPointsError, DegeneratePolynomialError) as e:
        continue

      #want this to be small
      nonzerolinearconstraint = result.linearconstraint[np.nonzero(result.linearconstraint)]
      figureofmerit = (result.fun >= 0), len(result.linearconstraint) - len(nonzerolinearconstraint), sum(np.log(abs(nonzerolinearconstraint))**2)

      if debugprint:
        print "---------------------------------"
        print result.linearconstraint
        print figureofmerit

      if best is None or figureofmerit < best[3]:
        best = permutation, permutationdict, result, figureofmerit
        if debugprint: print "new best"

      if debugprint: print "---------------------------------"

      signs[np.sign(result.fun)].append(permutation)
      if {
        "best": result.fun > 0 or figureofmerit <= (False, 0, 50),
        "asneeded": True,
        "best_gothroughall": False,
      }[permutationmode]:
        break

    if best is None:
      if "setsmallestcoefficientsto0" not in kwargs:
        kwargs["setsmallestcoefficientsto0"] = True
        return self.minimize_permutations(debugprint=debugprint, permutationmode=permutationmode, **kwargs)
      if "cmdlinestotry" not in kwargs:
        kwargs["cmdlinestotry"] = "smallparalleltdeg", "smallparalleltdegstepctrl", "smallparallel"#, "easy"
        return self.minimize_permutations(debugprint=debugprint, permutationmode=permutationmode, **kwargs)
      raise NoCriticalPointsError("Couldn't minimize polynomial under any permutation:\n{}".format(coeffs))

    permutation, permutationdict, result, figureofmerit = best

    #import pprint; pprint.pprint(signs)

    return result

  def minimize_permutationsasneeded(self, *args, **kwargs):
    return self.minimize_permutations(*args, permutationmode="asneeded", **kwargs)



class PolynomialBaseStandardLetters(PolynomialBase):
  @property
  def variableletters(self):
    return tuple(getnvariableletters(self.nvariables))

class ExplicitPolynomial(PolynomialBase):
  def __init__(self, monomials, variableletters, dehomogenizevariablesorder):
    self.__monomials = tuple(monomials)
    self.__variableletters = tuple(variableletters)
    if dehomogenizevariablesorder is not None: dehomogenizevariablesorder = tuple(dehomogenizevariablesorder)
    self.__dehomogenizevariablesorder = dehomogenizevariablesorder
    super(ExplicitPolynomial, self).__init__()
  @property
  def monomials(self): return self.__monomials
  @property
  def variableletters(self): return self.__variableletters
  @property
  def dehomogenizevariablesorder(self): return self.__dehomogenizevariablesorder

class PolynomialBaseProvideCoeffs(PolynomialBase):
  def __init__(self, coeffs):
    self.__coeffs = coeffs
    super(PolynomialBaseProvideCoeffs, self).__init__()

  @abc.abstractproperty
  def monomialswithoutcoeffs(self):
    """
    should be a list or generator of Counters, e.g. Counter({"x": 1, "y": 2} for the x*y^2 term,
    or something that can be passed as an argument to Counter, e.g. "xyy"
    __init__ will expect coefficients to be in the order corresponding to these terms
    """

  @property
  def monomials(self):
    for coeff, monomial in itertools.izip_longest(self.__coeffs, self.monomialswithoutcoeffs):
      if coeff is None or monomial is None:
        raise IndexError("Provided {} coefficients, need {}".format(len(self.__coeffs), len(list(self.monomialswithoutcoeffs))))
      yield Monomial(coeff, monomial)

class PolynomialNd(PolynomialBaseProvideCoeffs, PolynomialBaseStandardLetters):
  def __init__(self, d, n, coeffs):
    self.__degree = d
    self.__nvariables = n
    super(PolynomialNd, self).__init__(coeffs)
  @property
  def monomialswithoutcoeffs(self):
    xand1 = getnvariableletters(self.__nvariables+1)
    return itertools.combinations_with_replacement(xand1, self.__degree)
  @property
  def dehomogenizevariablesorder(self): return self.variableletters

class DoubleQuadratic(PolynomialBaseProvideCoeffs):
  def __init__(self, n1, n2, coeffs):
    self.__nvariables1 = n1
    self.__nvariables2 = n2
    self.__degree1 = self.__degree2 = 2 = 2
  @property
  def variables1(self):
    return getnvariableletters(self.__nvariables1, frombeginning=True)
  @property
  def variables2(self):
    return getnvariableletters(self.__nvariables2, frombeginning=True)
  @property
  def monomialswithoutcoeffs(self):
    return itertools.product(
      itertools.combinationwithreplacement(self.variables1, self.__degree1),
      itertools.combination_with_replacement(self.variables2, self.__degree2)
    )
  @property
  def dehomogenizevariablesorder(self):
    vv1 = iter(self.variables1)
    vv2 = iter(self.variables2)
    while True:
      try:
        v1 = next(vv1)
      except StopIteration:
        break
        pass
      try:
        v2 = next(vv2)
      except StopIteration:
        yield v2,
        break
      yield v1, v2
    for _ in v1: yield _,
    for _ in v2: yield _,

class DegeneratePolynomialError(ValueError):
  def __init__(self, polynomial, variable, power):
    super(DegeneratePolynomialError, self).__init__("Can't find the boundary polynomial of {} because it doesn't have a nonzero {}^{} term".format(polynomial, variable, power))

class NoCoeffGapError(ValueError): pass

class NoCriticalPointsError(ValueError):
  def __init__(self, coeffs, moremessage=None, solutions=None):
    message = "error finding critical points for polynomial: {}".format(coeffs)
    if moremessage: message += "\n\n"+moremessage
    super(NoCriticalPointsError, self).__init__(message)
    self.coeffs = coeffs
    self.solutions = solutions

def printresult(function):
  def newfunction(*args, **kwargs):
    result = function(*args, **kwargs)
    print args, kwargs
    print result
    raw_input()
    return result
  return newfunction

def permutations_differentonesfirst(iterable):
  """
  Variation of itertools.permutations:
    it tries to find the one that is most different from all the ones
    returned so far.  For example, permutations_finddifferentones("ABCD")
    starts by yielding ABCD as usual, but then goes to BADC
    followed by CDAB and DCBA
  """
  tpl = tuple(iterable)
  permutations = list(itertools.permutations(tpl))
  done = []
  while permutations:
    best = None
    bestfom = -float("inf")
    for permutation in permutations:
      figureofmerit = sum(sum(x != y for x, y in itertools.izip(permutation, oldpermutation)) for oldpermutation in done)
      if figureofmerit > bestfom:
        bestfom = figureofmerit
        best = permutation
    done.append(best)
    permutations.remove(best)
    yield best


if __name__ == "__main__":
  coeffs = np.array([float(_) for _ in """
        5.49334216e-09 -9.84400122e-09  4.38160058e-07 -1.26382479e-07
       -9.12089215e-10  1.01516341e-06 -1.26332021e-07  1.09256020e-07
        1.63314502e-09  1.03372119e-05 -4.96514478e-06 -7.80328090e-08
        5.99469948e-07  2.18440354e-08  9.52325483e-09 -5.08274877e-07
        6.57523730e-06  6.04471018e-06 -1.30160136e-06 -6.39638559e-07
        4.03655366e-07  1.97068383e-08  1.61606392e-07 -1.96627744e-08
        0.00000000e+00  1.38450711e-05  9.09015592e-06 -1.71721919e-06
       -9.80367599e-06  8.96773536e-07  8.06458361e-07  1.43873072e-06
       -1.19835291e-07  0.00000000e+00  0.00000000e+00  4.06317330e-07
       -3.33848064e-07 -2.14655695e-07  9.67377758e-08  2.86534044e-06
        1.28511390e-06 -8.78287961e-07  5.24488740e-06 -1.27394601e-06
        1.19881998e-05 -5.41396188e-07 -4.49905521e-07  1.26380292e-07
        4.44793949e-07 -7.02872007e-08  4.20198939e-16  4.65950944e-08
       -8.35958714e-09  2.03796222e-07  0.00000000e+00  9.50188566e-07
        3.69815932e-06 -1.11180898e-06  1.87621678e-06 -9.39022434e-07
        1.35841041e-05 -3.41416561e-06  7.81379717e-07  3.12984343e-14
        0.00000000e+00  8.04957702e-07 -1.34019321e-07  1.47580088e-06
        0.00000000e+00  0.00000000e+00
  """.split()])

  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--verbose", action="store_true")
  args = p.parse_args()

  print PolynomialNd(4, 4, coeffs).minimize_permutationsasneeded(verbose=True)

  #coeffs = coeffswithpermutedvariables(4, 4, coeffs, {"1": "z", "z": "1", "x": "x", "y": "y", "w": "w"})

  #print np.array(list(findcriticalpointspolynomialnd(4, 4, coeffs, **args.__dict__)))
