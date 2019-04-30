import re

from fileio import RootCd, RootFile
from templatecomponent import TemplateComponent

class Tree(object):
  def __init__(self, filename, treename, debug=False):
    self.__filename = filename
    self.__treename = treename
    self.__debug = debug
    self.__entered = False
    self.__iterated = False
    self.__templatecomponentargs = []
    self.__templatecomponents = []

  @property
  def filename(self): return self.__filename
  @property
  def treename(self): return self.__treename

  def __enter__(self):
    import ROOT

    f = self.__f = RootFile(self.__filename)
    f.__enter__()
    self.__t = getattr(f, self.__treename)
    self.__entered = True
    self.__t.SetBranchStatus("*", 0)

    for args, kwargs in self.__templatecomponentargs:
      self.maketemplatecomponent(*args, **kwargs)

    return self

  def __exit__(self, *errorstuff):
    self.__f.__exit__(*errorstuff)

  def registertemplatecomponent(self, *args, **kwargs):
    if self.__entered:
      raise RuntimeError("Can't add a template component after entering the tree")

    import ROOT

    kwargs["directory"] = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())
    self.__templatecomponentargs.append((args, kwargs))
    index = len(self.__templatecomponentargs)-1
    return lambda: self.__templatecomponents[index]

  def maketemplatecomponent(
    self, name, printprefix,
    xformula, xbins, xmin, xmax,
    yformula, ybins, ymin, ymax,
    zformula, zbins, zmin, zmax,
    cutformula, weightformula,
    mirrortype, scaleby, directory,
  ):
    import ROOT

    for formula in xformula, yformula, zformula, weightformula, cutformula:
      for branch in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", formula):
        self.__t.SetBranchStatus(branch, 1)
        if len(self):
          try:
            self.__t.GetEntry(0)
            getattr(self.__t, branch)
          except AttributeError:
            raise ValueError("Bad branch "+branch+" in "+str(self))

    with RootCd(directory):
      self.__templatecomponents.append(
        TemplateComponent(
          name, printprefix,
          ROOT.TTreeFormula(name+"_x", xformula, self.__t), xbins, xmin, xmax,
          ROOT.TTreeFormula(name+"_y", yformula, self.__t), ybins, ymin, ymax,
          ROOT.TTreeFormula(name+"_z", zformula, self.__t), zbins, zmin, zmax,
          ROOT.TTreeFormula(name+"_cut", cutformula, self.__t),
          ROOT.TTreeFormula(name+"_weight", weightformula, self.__t),
          mirrortype, scaleby,
        )
      )

  def __str__(self):
    return "{}:{}".format(self.__filename, self.__treename)

  def __len__(self):
    try:
      return self.__len
    except AttributeError:
      self.__len = self.__t.GetEntries()
      return len(self)

  def __iter__(self):
    if self.__iterated:
      raise ValueError("Already iterated through {}".format(self))
    self.__iterated = True

    print "Iterating through {} ({} entries)".format(self, self.__t.GetEntries())

    for i, entry in enumerate(self.__t, start=1):
      yield entry
      if i % 10000 == 0 or i == len(self):
        print "{} / {}".format(i, len(self))
        if self.__debug: break

  def fillall(self):
    print
    print "Filling:"
    for _ in self.__templatecomponents:
       print "  {:40} {:45}".format(_.printprefix, _.name)
    print
    for entry in self:
      for _ in self.__templatecomponents:
        _.fill()
    print
    print "Integrals:"
    for _ in self.__templatecomponents:
       print "  {:40} {:45} {:10.3e}".format(_.printprefix, _.name, _.integral)
    print
