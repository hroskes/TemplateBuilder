import abc, array, itertools, os

from collections import Counter

import numpy as np
import uncertainties

from moremath import weightedaverage
from templatecomponent import TemplateComponent

class TemplateBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, name, printprefix, mirrortype, scaleby, floor, xbins, ybins, zbins, templatecomponents):
    self.__name = name
    self.__printprefix = printprefix
    self.__mirrortype = mirrortype
    self.__scaleby = scaleby
    self.__floor = floor
    self.__xbins = xbins
    self.__ybins = ybins
    self.__zbins = zbins
    self.__templatecomponents = templatecomponents

    if mirrortype not in (None, "symmetric", "antisymmetric"):
      raise ValueError("invalid mirrortype {}: has to be None, symmetric, or antisymmetric".format(mirrortype))

  @property
  def name(self): return self.__name
  @property
  def printprefix(self): return self.__printprefix
  @property
  def mirrortype(self): return self.__mirrortype
  @property
  def scaleby(self): return self.__scaleby
  @property
  def floor(self): return self.__floor

  @property
  def xbins(self): return self.__xbins
  @property
  def ybins(self): return self.__ybins
  @property
  def zbins(self): return self.__zbins
  @property
  def binsxyz(self):
    return itertools.product(xrange(1, self.xbins+1), xrange(1, self.ybins+1), xrange(1, self.zbins+1))

  @property
  def templatecomponents(self):
    return self.__templatecomponents

class Template(TemplateBase):
  """
  Structure:
    A Template is the sum of multiple TemplateComponents.
    Each TemplateComponent is made of one or more TemplateComponentPieces,
    which are independent estimates of the TemplateComponent's bin content.
    E.g. a VH Template could be made of ZH and WH TemplateComponents,
    each of which is reweighted from a number of ZH or WH generated samples.
  """
  def __init__(
    self, name, printprefix, trees,
    xformula, xbins, xmin, xmax,
    yformula, ybins, ymin, ymax,
    zformula, zbins, zmin, zmax,
    cutformula, weightformulas,
    mirrortype, scaleby, floor,
    reuseifexists=False,
  ):
    filenames = [tree.filename for listoftrees in trees for tree in listoftrees]
    commonprefix = os.path.commonprefix(filenames)
    commonsuffix = os.path.commonprefix(list(_[::-1] for _ in filenames))[::-1]
    assert commonprefix + "plain" + commonsuffix not in filenames

    if scaleby is None: scaleby = 1

    import ROOT

    self.__tdirectory = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())

    hkey = self.__tdirectory.FindKey(name)
    if reuseifexists and hkey:
      self.__alreadyexists = True
      self.__h = hkey.ReadObj()
      self.__finalized = self.__didscale = self.__didcheckmirror = self.__didfloor = True
      self.__h.SetDirectory(0)
    else:
      self.__alreadyexists = False
      self.__h = ROOT.TH3F(
        name, name,
        xbins, xmin, xmax,
        ybins, ymin, ymax,
        zbins, zmin, zmax,
      )

      self.__h.SetDirectory(0)

      self.__finalized = self.__didscale = self.__didcheckmirror = self.__didfloor = False

    subdirectories = {
      tree.filename: tree.filename.replace(commonprefix, "", 1)[::-1].replace(commonsuffix[::-1], "", 1)[::-1]
      for listoftrees in trees for tree in listoftrees
    }
    assert "plain" not in subdirectories.values()
    for k, v in subdirectories.items():
      if not v:
        subdirectories[k] = "plain"

    templatecomponents = [
      TemplateComponent(
        name, listoftrees, [subdirectories[tree.filename] for tree in listoftrees],
        printprefix,
        xformula, xbins, xmin, xmax,
        yformula, ybins, ymin, ymax,
        zformula, zbins, zmin, zmax,
        cutformula, weightformula,
        mirrortype, scaleby,
      ) for listoftrees, weightformula in itertools.izip_longest(trees, weightformulas)
    ]
    super(Template, self).__init__(name, printprefix, mirrortype, scaleby, floor, xbins, ybins, zbins, templatecomponents)

  def GetBinContentError(self, *args):
    return uncertainties.ufloat(self.__h.GetBinContent(*args), self.__h.GetBinError(*args))

  def SetBinContentError(self, *args):
    if self.finalized:
      raise RuntimeError("Can't set bin content after it's finalized")
    self.__h.SetBinContent(*args[:-1]+(uncertainties.nominal_value(args[-1]),))
    self.__h.SetBinError(*args[:-1]+(uncertainties.std_dev(args[-1]),))

  @property
  def integral(self):
    return sum(self.GetBinContentError(x, y, z) for x, y, z in self.binsxyz)

  def doscale(self):
    if self.__didscale: raise RuntimeError("Trying to scale twice!")
    self.__didscale = True

  def checkmirror(self):
    if self.__didcheckmirror: raise RuntimeError("Trying to mirror twice!")
    self.__didcheckmirror = True
    if self.mirrortype is None: return
    sign = {"symmetric": 1, "antisymmetric": -1}[self.mirrortype]
    for x, y, z in self.binsxyz:
      if y > self.ybins / 2: continue
      othery = self.ybins+1-y
      if (
        self.GetBinContentError(x, y, z).n == sign*self.GetBinContentError(x, othery, z).n
        and self.GetBinContentError(x, y, z).s != self.GetBinContentError(x, othery, z).s
        and np.isclose(self.GetBinContentError(x, y, z).s, self.GetBinContentError(x, othery, z).s, atol=0, rtol=1e-15)
      ):
        self.SetBinContentError(x, othery, z, self.GetBinContentError(x, y, z))
      if (self.GetBinContentError(x, y, z).n, self.GetBinContentError(x, y, z).s) != (sign*self.GetBinContentError(x, othery, z).n, self.GetBinContentError(x, othery, z).s):
        raise RuntimeError("Mirror didn't happen: ({}, {}, {}) {!r} {!r}".format(x, y, z, self.GetBinContentError(x, y, z), self.GetBinContentError(x, othery, z)))

  def dofloor(self):
    if self.__didfloor: raise RuntimeError("Trying to floor twice!")
    self.__didfloor = True
    floor = self.floor
    if floor is None: return
    if floor.nominal_value <= 0:
      raise ValueError("Invalid floor {}: has to be positive.".format(floor.nominal_value))

    if floor.std_dev == 0 and any(self.__h.GetBinContent(x, y, z) for x, y, z in self.binsxyz):
      #use this procedure to estimate the error for floored bins
      maxerrorratio, errortoset = max(
        (self.__h.GetBinError(x, y, z) / self.__h.GetBinContent(x, y, z), self.__h.GetBinError(x, y, z))
          for x, y, z in self.binsxyz
        if self.__h.GetBinContent(x, y, z) != 0
      )
      #the reasoning being that if there's a bin with just one entry 2.3 +/- 2.3, then the zero bin could also have 2.3
      #but we can't draw that conclusion from a bin 1000 +/- 5.5

      floor = uncertainties.ufloat(floor.nominal_value, errortoset)

    for x, y, z in self.binsxyz:
      if self.__h.GetBinContent(x, y, z) <= floor.nominal_value:
        self.__h.SetBinContent(x, y, z, floor.nominal_value)
        self.__h.SetBinError(x, y, z, floor.std_dev)

  def finalize(self):
    if self.__finalized:
      assert self.__didscale
      assert self.__didfloor
      assert self.__didcheckmirror
    self.doscale()
    self.dofloor()
    self.checkmirror()
    self.__finalized = True
    self.__h.SetDirectory(self.__tdirectory)

  @property
  def finalized(self):
    return self.__finalized

  @property
  def alreadyexists(self):
    return self.__alreadyexists

  @property
  def rootless(self):
    kwargs = {thing: getattr(self, thing) for thing in ("name", "printprefix", "mirrortype", "scaleby", "floor", "xbins", "ybins", "zbins", "templatecomponents")}
    return RootlessTemplate(**kwargs)

class RootlessTemplate(TemplateBase):
  def __init__(self, **kwargs):
    kwargs["templatecomponents"] = [_.rootless for _ in kwargs["templatecomponents"]]
    super(RootlessTemplate, self).__init__(**kwargs)
