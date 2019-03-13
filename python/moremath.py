#!/usr/bin/env python

import autograd.numpy as np
import uncertainties

def weightedaverage(values):
  values = tuple(values)
  if not values: raise IOError("Can't take the weighted average of an empty array")
  return uncertainties.ufloat(
    sum(x.nominal_value / x.std_dev**2 for x in values) / sum(1 / x.std_dev**2 for x in values),
    sum(1 / x.std_dev**2 for x in values) ** -0.5
  )

def cubicformula(coeffs):
  a, b, c, d = coeffs

  Delta0 = b**2 - 3*a*c
  Delta1 = 2*b**3 - 9*a*b*c + 27*a**2*d
  C = ((Delta1 + (1 if Delta1>0 else -1) * (Delta1**2 - 4*Delta0**3 + 0j)**0.5) / 2) ** (1./3)

  xi = 0.5 * (-1 + (3**0.5)*1j)

  return np.array([-1/(3.*a) * (b + xi**k*C + Delta0/(xi**k * C)) for k in range(3)])

def minimizequartic(coeffs):
  a, b, c, d, e = coeffs
  if a < 0: return -float("inf")
  flatpoints = cubicformula(np.array([4*a, 3*b, 2*c, d]))
  result = float("inf")
  for x in flatpoints:
    if abs(np.imag(x)) > 1e-12: continue
    x = np.real(x)
    result = min(result, a*x**4 + b*x**3 + c*x**2 + d*x + e)
  return np.real(result)

if __name__ == "__main__":
  from autograd import holomorphic_grad, grad, jacobian, hessian, linear_combination_of_hessians
  print cubicformula(np.array([1., 2., 3., 4.]))

  def cubicformula0(coeffs): return cubicformula(coeffs)[0]

  epsilonvalue = 1e-6

  coeffs = np.array([1., 2., 3., 3., 5.])
  def epsilon(i): return np.array([epsilonvalue if _==i else 0 for _ in range(5)])

  for i in range(5):
    print (minimizequartic(coeffs+epsilon(i)) - minimizequartic(coeffs-epsilon(i))) / (2*epsilonvalue)
  print jacobian(minimizequartic)(coeffs)
  print
  print
  print hessian(minimizequartic)(coeffs)
  print linear_combination_of_hessians(minimizequartic)(coeffs, np.array([1,2,3,4,5.]))