from __future__ import print_function

import abc, copy, functools, itertools, logging, multiprocessing, textwrap, sys, traceback

import numpy as np
from scipy import optimize
from uncertainties import nominal_value, std_dev, ufloat

from cuttingplanemethod import cuttingplanemethod1dquadratic, cuttingplanemethod1dquartic, cuttingplanemethod3dquadratic, cuttingplanemethod4dquadratic, cuttingplanemethod4dquartic, cuttingplanemethod4dquartic_1stvariableonlyeven, cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_1stvariableonlyeven, cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic_1stvariableonlyeven, cuttingplanemethod4dquartic_4thvariablezerocubic_1stvariableonlyeven, cuttingplanemethod4dquartic_4thvariablequadratic, cuttingplanemethod4dquartic_4thvariablequadratic_1stvariableonlyeven, cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic, cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic, cuttingplanemethod4dquartic_4thvariablezerocubic
from moremath import kspoissongaussian, weightedaverage
from optimizeresult import OptimizeResult
from polynomialalgebra import NoCriticalPointsError

class BadFitStatusException(Exception):
  def __init__(self, fitresult):
    self.fitresult = fitresult
    super(BadFitStatusException, self).__init__("Fit returned with a status indicating failure:\n\n"+str(fitresult))

def ConstrainedTemplates(constrainttype, *args, **kwargs):
  return {
    "unconstrained": OneTemplate,
    "oneparameterHVV": OneParameterHVV,
    "oneparameterVVHVV": OneParameterVVHVV,
    "threeparameterHVV": ThreeParameterHVV,
    "twobythreeparameterggHVV": TwoByThreeParameterggHVV,
    "fourparameterHVV": FourParameterHVV,
    "fourparameterVVHVV": FourParameterVVHVV,
    "fourparameterWWHVV": FourParameterWWHVV,
    "fourparameterVVHVV_nog4int": FourParameterVVHVV_nog4int,
    "fourparameterWWHVV_nog4int": FourParameterWWHVV_nog4int,
  }[constrainttype](*args, **kwargs)

class ConstrainedTemplatesBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, templates, logfile=None, nthreads=1):
    self.__templates = templates
    if len(templates) != self.ntemplates:
      raise ValueError("Wrong number of templates ({}) for {}, should be {}".format(len(templates), type(self).__name__, self.ntemplates))

    self.__loggername = "constrainedtemplates"+templates[0].name
    logger = logging.getLogger(self.__loggername)
    logger.addHandler(logging.StreamHandler())
    if logfile is not None:
      logger.addHandler(logging.StreamHandler(logfile))
    logger.setLevel(logging.INFO)

    self.__nthreads = nthreads

  @property
  def templates(self): 
    return self.__templates

  @property
  def ntemplates(self): return len(self.templatenames)

  @abc.abstractproperty
  def templatenames(self): "can be a class member, names of the templates in order (e.g. SM, int, BSM)"

  @property
  def binsxyz(self):
    for binxyz in itertools.izip(*(t.binsxyz for t in self.templates)):
      binxyz = set(binxyz)
      if len(binxyz) != 1:
        raise ValueError("Templates have inconsistent binning")
      yield binxyz.pop()

  @property
  def xbins(self):
    result = {t.xbins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  @property
  def ybins(self):
    result = {t.ybins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  @property
  def zbins(self):
    result = {t.zbins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  def isbinsmall(self, x, y, z):
    return False

  def getcomponentbincontents(self, x, y, z):
      bincontents = []

      for template in self.templates:
        componentbincontents = []
        bincontents.append(componentbincontents)
        for component in template.templatecomponents:
          thisonescontent = {}
          componentbincontents.append(thisonescontent)
          for piece in component.templatecomponentpieces:
            thisonescontent[piece.name.replace(template.name+"_", "")] = piece.GetBinContentError(x, y, z)

      return bincontents

  def getcomponentbincontentsabs(self, x, y, z):
      bincontents = []

      for template in self.templates:
        componentbincontents = []
        bincontents.append(componentbincontents)
        for component in template.templatecomponents:
          thisonescontent = {}
          componentbincontents.append(thisonescontent)
          for piece in component.templatecomponentpieces:
            thisonescontent[piece.name.replace(template.name+"_", "")] = piece.GetBinContentErrorAbs(x, y, z)

      return bincontents

  def getcomponentsumsofallweights(self):
      sumofallweights = []

      for template in self.templates:
        componentsumofallweights = []
        sumofallweights.append(componentsumofallweights)
        for component in template.templatecomponents:
          thisonescontent = {}
          componentsumofallweights.append(thisonescontent)
          for piece in component.templatecomponentpieces:
            thisonescontent[piece.name.replace(template.name+"_", "")] = piece.sumofallweights

      return sumofallweights

  def write(self, thing):
    logger = logging.getLogger(self.__loggername)
    logger.info(str(thing))

  def makefinaltemplates(self, printbins, printallbins, binsortkey=lambda xyz: xyz):
    if all(_.alreadyexists for _ in self.templates):
      for template in self.templates:
        for component in template.templatecomponents:
          for piece in component.templatecomponentpieces:
            piece.lock()
      return
    assert not any(_.alreadyexists for _ in self.templates)

    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    self.write("Making the final templates:")
    for name, _ in itertools.izip(self.templatenames, self.templates):
      self.write("  {:>10}: {:40} {:45}".format(name, _.printprefix, _.name))
    self.write("from individual templates with integrals:")

    for template in self.templates:
      for component in template.templatecomponents:
        for piece in component.templatecomponentpieces:
          piece.lock()
          self.write("  {:45} {:10.3e}".format(piece.name, piece.integral))

    printmessage = {}
    warnings = {}
    finalbincontents = {}

    def binsortkeywithmirror(xyz):
      x, y, z = xyz
      if not any(t.mirrortype for t in self.__templates):
        return binsortkey(xyz)
      if not all(t.mirrortype for t in self.__templates):
        raise ValueError("Some templates are mirrored, some aren't")

      thiskey = binsortkey(xyz)

      mirrory = self.ybins - y + 1
      mirrorxyz = x, mirrory, z
      mirrorkey = binsortkey(mirrorxyz)

      return (
        min(thiskey, mirrorkey),   #do both xyz and mirrorxyz when the default sortkey would give the first of them
        min(xyz, mirrorxyz),       #if there are multiple sets of xyz with the same key, make sure these two are next to each other
        thiskey,                   #of xyz and mirrorxyz, choose the one with the smaller key first
      )

    xyzs = sorted(self.binsxyz, key=binsortkeywithmirror)

    if self.__nthreads > 1:
      pool = multiprocessing.Pool(processes=self.__nthreads)
      mapargs = [[x, y, z] for x, y, z in xyzs]
      selffindbincontentswrapper = functools.partial(findbincontentswrapper, self, printallbins=printallbins)

      bkptemplates = self.__templates
      self.__templates = [t.rootless for t in self.__templates]
      mapresult = pool.map(selffindbincontentswrapper, mapargs, chunksize=2)
      self.__templates = bkptemplates

      for xyz, findbincontentsresult in itertools.izip_longest(xyzs, mapresult):
        finalbincontents[xyz], printmessage[xyz], warnings[xyz] = findbincontentsresult

    else:
      for xyz in xyzs:
        finalbincontents[xyz], printmessage[xyz], warnings[xyz] = self.findbincontents(*xyz, printallbins=printallbins)

    for (x, y, z), bincontents in finalbincontents.iteritems():
      for t, content in itertools.izip(self.templates, bincontents):
        t.SetBinContentError(x, y, z, content)

    if printbins:
      self.write("")
      self.write("Bins you requested to print:")
      for xyz in printbins: self.write(printmessage[xyz])

    if warnings:
      self.write("")
      self.write("Warnings:")
      for xyz in self.binsxyz:
        if warnings[xyz]:
          self.write(warnings[xyz])

    for _ in self.templates:
      _.finalize()

    self.write("")
    self.write("final integrals:")
    for name, t in itertools.izip(self.templatenames, self.templates):
      self.write("  {:>10} = {:10.3e}".format(name, t.integral))
    self.write("")


  def findbincontents(self, x, y, z, printallbins=False):
    bincontents = self.getcomponentbincontents(x, y, z)
    bincontentsabs = self.getcomponentbincontentsabs(x, y, z)
    originalbincontents = copy.deepcopy(bincontents)

    warning = []

    for i, (componentbincontents, componentbincontentsabs, t) in enumerate(
      itertools.izip_longest(
        bincontents, bincontentsabs, self.templates
      )
    ):
      outlierwarning = []
      for bincontent, bincontentabs in itertools.izip(componentbincontents, componentbincontentsabs):
        outliers = self.findoutliers(bincontent, bincontentabs)
        bincontent.update(outliers)
        outlierwarning += list(outliers)
      if outlierwarning:
        warning.append("some errors have been inflated for "+t.name+": "+", ".join(sorted(outlierwarning)))

    printmessage = ""
    fmt1 = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
    fmt2 = fmt1 + " (was +/-{:10.3e})"
    fmt3 = fmt1 + "                     (sum(abs(wt)) {:10.3e})"
    fmt4 = fmt2 + " (sum(abs(wt)) {:10.3e})"
    for i, (t, thisonescontent, originalcontent, bincontentabs) in enumerate(itertools.izip(self.templates, bincontents, originalbincontents, bincontentsabs)):
      printmessage += "\n"+t.name+":"
      for componentcontent, componentoriginalcontent, componentbincontentabs in itertools.izip(thisonescontent, originalcontent, bincontentabs):
        for name, content in sorted(componentcontent.iteritems()):
          fmt = {
            (True, True): fmt1,
            (False, True): fmt2,
            (True, False): fmt3,
            (False, False): fmt4,
          }[content.n == componentoriginalcontent[name].n and content.s == componentoriginalcontent[name].s, i in self.pureindices]
          fmtargs = [name, content]
          if fmt in (fmt2, fmt4): fmtargs.append(componentoriginalcontent[name].s)
          if fmt in (fmt3, fmt4): fmtargs.append(componentbincontentabs[name].n)
          printmessage += "\n"+fmt.format(*fmtargs)

    issmall = self.isbinsmall(x, y, z)

    try:
      finalbincontents, fitprintmessage, fitwarning = self.computefinalbincontents(bincontents, bincontentsabs, issmall=issmall, logprefix="{:3} {:3} {:3}".format(x, y, z))
      if fitprintmessage: printmessage += "\n\n" + fitprintmessage.lstrip("\n")
      warning += fitwarning
    except BaseException as e:
      print("Error when finding content for bin", x, y, z)
      print(printmessage)
      if hasattr(e, "printmessage"): print(e.printmessage)
      raise

    for name, content in itertools.izip(self.templatenames, finalbincontents):
      printmessage += "\n\n"+fmt1.format("final "+name, content)

    printmessage = "  {:3d} {:3d} {:3d}:\n".format(x, y, z) + printmessage
    if printallbins:
      self.write(printmessage)
    else:
      self.write("  {:3d} {:3d} {:3d}".format(x, y, z))

    if warning or printallbins:
      if isinstance(warning, basestring):
        warning = [warning]
      else:
        warning = list(warning)
      warning = (
        "\n      ".join(
          ["  {:3d} {:3d} {:3d}:".format(x, y, z)]
          +warning
        )
      )

    return finalbincontents, printmessage, warning

  @abc.abstractmethod
  def computefinalbincontents(self, bincontents, bincontentsabs, issmall=False): pass

  def applymirrortoarray(self, array):
    if len(self.templates) != len(array):
      raise ValueError("array should have length {}".format(len(self.templates)))
    try:
      return np.array([
        {"symmetric": 1, "antisymmetric": -1}[t.mirrortype] * s
           if t.scaleby != 0
           else 0 * s
        for t, s in itertools.izip(self.templates, array)
      ])
    except KeyError:
      raise ValueError("Not all templates have mirror: " + ", ".join(str(t.mirrortype) for t in self.templates))

  def findoutliers(self, bincontent, bincontentabs, debugprint=False):
    bincontent = bincontent.copy()
    bincontentabs = bincontentabs.copy()

    relativeerror = {name: contentabs.s / contentabs.n if contentabs.n else float("inf") for name, contentabs in bincontentabs.iteritems()}
    outliers = {}

    for name in bincontent:
      if debugprint: print(name, bincontent[name], relativeerror[name], bincontentabs[name])
      if debugprint: print("still here!")
      errortoset = None
      for othername in bincontent:
        #look at the other names that have bigger errors but comparable relative errors
        if bincontentabs[othername].s <= bincontentabs[name].s: continue
        if bincontentabs[name].s == 0: continue
        if debugprint: print("here with", othername)
        if relativeerror[othername] <= relativeerror[name] * (
          (1 + 1.5 * np.log10(bincontentabs[othername].s / bincontentabs[name].s) * kspoissongaussian(1/relativeerror[name]**2))
        ):
          if debugprint: print("here 2 with", othername)
          if errortoset is None: errortoset = 0
          errortoset = max(errortoset, bincontentabs[othername].s)
      if errortoset is not None:
        outliers[name] = ufloat(bincontent[name].n, errortoset)

    return outliers

  def purebincontents(self, x, y, z):
    for t in self.puretemplates:
      yield t.GetBinContentError(x, y, z)

  @property
  def puretemplates(self):
    return [self.templates[i] for i in self.pureindices]

class OneTemplate(ConstrainedTemplatesBase):
  templatenames = "",
  pureindices = 0,

  def computefinalbincontents(self, bincontents, bincontentsabs, issmall=False, logprefix=None):
    printmessage = ""
    warning = []

    finalbincontent = 0
    #these will be the final estimates of the bin contents, with their errors
    for i, (componentbincontents, componentbincontentsabs) in enumerate(
      itertools.izip_longest(
        bincontents, bincontentsabs
      )
    ):
      for bincontent, bincontentabs in itertools.izip_longest(componentbincontents, componentbincontentsabs):
        if any(_.n for _ in bincontent.itervalues()):
          finalbincontent += weightedaverage(bincontent.itervalues())

    return [finalbincontent], printmessage, warning

class ConstrainedTemplatesWithFit(ConstrainedTemplatesBase):
  def computefinalbincontents(self, bincontents, bincontentsabs, issmall=False, logprefix=None):
    warning = []

    #Each template component piece produces a 3D probability distribution in (SM, int, BSM)
    #can improve this by including correlations and/or by not approximating as Gaussian

    #want to put the constraint in the following form:
    # likelihood = 1/2 x^T Q x + c^T x + r

    contentswitherrors = [0 for t in self.templates]
    #these will be the final estimates of the bin contents, with their errors
    for i, (componentbincontents, componentbincontentsabs, t) in enumerate(
      itertools.izip_longest(
        bincontents, bincontentsabs, self.templates
      )
    ):
      for bincontent, bincontentabs in itertools.izip_longest(componentbincontents, componentbincontentsabs):
        if any(_.n for _ in bincontent.itervalues()):
          contentswitherrors[i] += weightedaverage(bincontent.itervalues())

    x0 = np.array([nominal_value(content) for content in contentswitherrors])
    sigma = np.array([std_dev(content) for content in contentswitherrors])

    if np.all(x0 == 0) and np.all(sigma == 0):
      finalbincontents = np.array([0]*self.ntemplates)
      fitprintmessage = "all templates have zero content for this bin"
    else:
      cachekey = tuple(x0), tuple(sigma)
      if all(t.mirrortype for t in self.templates):
        mirroredx0 = self.applymirrortoarray(x0)
        mirroredcachekey = tuple(mirroredx0), tuple(sigma)
      if cachekey not in self.__fitresultscache:
        fitresult = self.__fitresultscache[cachekey] = self.docuttingplanes(
          x0,
          sigma,
          issmall=issmall,
          logprefix=logprefix,
        )
        if all(t.mirrortype for t in self.templates):
          self.__fitresultscache[mirroredcachekey] = OptimizeResult(
            x=self.applymirrortoarray(fitresult.x),
            fun=fitresult.fun,
            message="(mirrored) "+fitresult.message,
            status=fitresult.status,
            nit=fitresult.nit,
            maxcv=fitresult.maxcv
          )
      fitresult = self.__fitresultscache[cachekey]

      if fitresult.maxcv:
        raise ValueError("Fit failed with constraint violation\n\n{}".format(fitresult))

      finalbincontents = fitresult.x

      if fitresult.nit == 1:
        fitprintmessage = "global minimum already satisfies constraint"
      else:
        fitprintmessage = str(fitresult)
        warning.append("fit converged in {0.nit} with NLL = {0.fun}".format(fitresult))
        warning.append(fitresult.message)

    return finalbincontents, fitprintmessage, warning

  def __init__(self, *args, **kwargs):
    super(ConstrainedTemplatesWithFit, self).__init__(*args, **kwargs)
    self.__fitresultscache = {}

  @abc.abstractproperty
  def pureindices(self): "can be a class member"
  @abc.abstractmethod
  def cuttingplanefunction(self, x0, sigma, maxfractionaladjustment): "can be static"
  @abc.abstractproperty
  def cuttingplanehaspermutations(self):
    """
    can be a class member
    say if the cutting plane function has a usepermutations kwarg
    """

  def docuttingplanes(self, x0, sigma, maxfractionaladjustment=1e-6, maxiter=None, issmall=False, **kwargs):
    if maxiter is None: maxiter = self.defaultmaxiter
    if issmall: maxiter /= 10
    if all(x0[i] == 0 for i in xrange(self.ntemplates) if i not in self.pureindices):
      return OptimizeResult(
        x=x0,
        success=True,
        status=1,
        maxcv=0,
        fun=0,
        nit=1,
        message="all interference terms are 0",
      )
    try:
      result = self.cuttingplanefunction(x0, sigma, maxfractionaladjustment=maxfractionaladjustment, maxiter=maxiter, **kwargs)
      if result.status >= 3: raise BadFitStatusException(result)
      return result
    except (BadFitStatusException, NoCriticalPointsError) as e:
      if isinstance(e, BadFitStatusException) and hasattr(e.fitresult, "error_step2"):  #then it's a 2 step fit and the first went fine
        return e.fitresult
      if issmall:
        result = OptimizeResult(
          x=np.zeros(len(x0)),
          success=True,
          status=2,
          maxcv=0,
          fun="N/A"
        )
        if isinstance(e, BadFitStatusException):
          result.update(
            failedresult=e.fitresult,
            nit=e.fitresult.nit,
            message=e.fitresult.message,
          )
        elif isinstance(e, NoCriticalPointsError):
          result.update(
            nit=-1,
            message=str(e),
          )
        else:
          assert False, e
        result.message += "\nThis is a small bin, so set the contents to 0 everywhere"
        return result

      if self.cuttingplanehaspermutations:
        try:
          result = self.cuttingplanefunction(x0, sigma, maxfractionaladjustment=maxfractionaladjustment, maxiter=maxiter, usepermutations=True, **kwargs)
          if result.status >= 3: raise BadFitStatusException(result)
          return result
        except BadFitStatusException as e:
          if isinstance(e, BadFitStatusException) and hasattr(e.fitresult, "error_step2"):  #then it's a 2 step fit and the first went fine
            return e.fitresult
      raise

  def isbinsmall(self, x, y, z):
    for index in self.pureindices:
      bincontent = sum(
        weightedaverage(templatecomponentbincontent.itervalues())
        for templatecomponentbincontent in self.getcomponentbincontents(x, y, z)[index]
      )
      integral = sum(
        weightedaverage(templatecomponentsumofallweights.itervalues())
        for templatecomponentsumofallweights in self.getcomponentsumsofallweights()[index]
      )
      if bincontent > integral / 50000:
        return False
    return True

  defaultmaxiter = 2000

class OneParameterHVV(ConstrainedTemplatesWithFit):
  templatenames = "SM", "int", "BSM"
  pureindices = 0, 2
  cuttingplanefunction = staticmethod(cuttingplanemethod1dquadratic)
  cuttingplanehaspermutations = False
  defaultmaxiter = 2000

class OneParameterVVHVV(ConstrainedTemplatesWithFit):
  templatenames = "SM", "g13gi1", "g12gi2", "g11gi3", "BSM"
  pureindices = 0, 4
  cuttingplanefunction = staticmethod(cuttingplanemethod1dquartic)
  cuttingplanehaspermutations = False

class ThreeParameterHVV(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM", "g11gj1", "g11gk1", "g11gl1",
    "j",  "gj1gk1", "gj1gl1",
    "k",  "gk1gl1",
    "l",
  )
  pureindices = 0, 4, 7, 9
  cuttingplanefunction = staticmethod(cuttingplanemethod3dquadratic)
  cuttingplanehaspermutations = True
  defaultmaxiter = 2000

class TwoByThreeParameterggHVV(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM_SM", "SM_g11gj1", "SM_g11gk1", "SM_g11gl1",
    "SM_j",  "SM_gj1gk1", "SM_gj1gl1",
    "SM_k",  "SM_gk1gl1",
    "SM_l",

    "G1GI_SM", "G1GI_g11gj1", "G1GI_g11gk1", "G1GI_g11gl1",
    "G1GI_j",  "G1GI_gj1gk1", "G1GI_gj1gl1",
    "G1GI_k",  "G1GI_gk1gl1",
    "G1GI_l",

    "GI_SM", "GI_g11gj1", "GI_g11gk1", "GI_g11gl1",
    "GI_j",  "GI_gj1gk1", "GI_gj1gl1",
    "GI_k",  "GI_gk1gl1",
    "GI_l",
  )
  pureindices = ()
  cuttingplanefunction = staticmethod(cuttingplanemethod2dquadraticby3dquadratic)
  cuttingplanehaspermutations = True
  defaultmaxiter = 2000

class FourParameterHVV(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM", "g11gi1", "g11gj1", "g11gk1", "g11gl1",
    "i",  "gi1gj1", "gi1gk1", "gi1gl1",
    "j",  "gj1gk1", "gj1gl1",
    "k",  "gk1gl1",
    "l",
  )
  pureindices = 0, 5, 9, 12, 14
  cuttingplanefunction = staticmethod(cuttingplanemethod4dquadratic)
  cuttingplanehaspermutations = True
  defaultmaxiter = 2000

class FourParameterVVHVV(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM",
    "g13gi1", "g13gj1",    "g13gk1",       "g13gl1",

    "g12gi2", "g12gi1gj1", "g12gi1gk1",    "g12gi1gl1",
              "g12gj2",    "g12gj1gk1",    "g12gj1gl1",
                           "g12gk2",       "g12gk1gl1",
                                           "g12gl2",

    "g11gi3", "g11gi2gj1", "g11gi2gk1",    "g11gi2gl1",
              "g11gi1gj2", "g11gi1gj1gk1", "g11gi1gj1gl1",
                           "g11gi1gk2",    "g11gi1gk1gl1",
                                           "g11gi1gl2",
              "g11gj3",    "g11gj2gk1",    "g11gj2gl1",
                           "g11gj1gk2",    "g11gj1gk1gl1",
                                           "g11gj1gl2",
                           "g11gk3",       "g11gk2gl1",
                                           "g11gk1gl2",
                                           "g11gl3",

    "i",
              "gi3gj1",    "gi3gk1",       "gi3gl1",

              "gi2gj2",    "gi2gj1gk1",    "gi2gj1gl1",
                           "gi2gk2",       "gi2gk1gl1",
                                           "gi2gl2",

              "gi1gj3",    "gi1gj2gk1",    "gi1gj2gl1",
                           "gi1gj1gk2",    "gi1gj1gk1gl1",
                                           "gi1gj1gl2",
                           "gi1gk3",       "gi1gk2gl1",
                                           "gi1gk1gl2",
                                           "gi1gl3",

    "j",
                           "gj3gk1",       "gj3gl1",

                           "gj2gk2",       "gj2gk1gl1",
                                           "gj2gl2",

                           "gj1gk3",       "gj1gk2gl1",
                                           "gj1gk1gl2",
                                           "gj1gl3",

    "k",
                                           "gk3gl1",

                                           "gk2gl2",

                                           "gk1gl3",

    "l",

  )
  pureindices = 0, 35, 55, 65, 69
  def cuttingplanefunction(self, x0, sigma, *args, **kwargs):
    if np.all(x0[self.gZ34indices,] == 0): #this happens for VBF when there are no reweighted ZZ fusion events in the bin
      return cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic(x0, sigma, *args, **kwargs)

    elif np.all(x0[self.gZ3indices,] == 0): #this happens for VBF when there are no reweighted ZZ fusion events in the bin but there are events in the L1Zg sample
      return cuttingplanemethod4dquartic_4thvariablezerocubic(x0, sigma, *args, **kwargs)

    elif max(
      ufloat(x0[i], sigma[i])
      for i in range(self.ntemplates)
      if i in self.gZ34indices and i in self.pureindices
    ) / min(
      ufloat(x0[i], sigma[i])
      for i in range(self.ntemplates)
      if i not in self.gZ34indices and i in self.pureindices
    ) < 1e-3:
      return cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic(x0, sigma, *args, **kwargs)

    else:
      return cuttingplanemethod4dquartic(x0, sigma, *args, **kwargs)
  gZ3indices = tuple(i for i, _ in enumerate(templatenames) if "gl3" in _)
  gZ34indices = tuple(i for i, _ in enumerate(templatenames) if "gl3" in _ or _ == "l")
  cuttingplanehaspermutations = True

class FourParameterWWHVV(ConstrainedTemplatesWithFit):
  #https://stackoverflow.com/q/13905741/5228524
  templatenames = tuple(name for i, name in enumerate(FourParameterVVHVV.templatenames) if i not in FourParameterVVHVV.gZ34indices)
  pureindices = tuple(
    index - sum(1 for i in FourParameterVVHVV.gZ34indices if i < index)
    for index in FourParameterVVHVV.pureindices
    if index not in FourParameterVVHVV.gZ34indices
  )
  cuttingplanefunction = staticmethod(cuttingplanemethod4dquartic_4thvariablequadratic)
  cuttingplanehaspermutations = True

class FourParameterVVHVV_nog4int(ConstrainedTemplatesWithFit):
  templatenames = tuple(name for name in FourParameterVVHVV.templatenames if "gi1" not in name and "gi3" not in name)
  pureindices = tuple(
    index - sum(1 for i, name in enumerate(FourParameterVVHVV.templatenames) if i < index and ("gi1" in name or "gi3" in name))
    for index in FourParameterVVHVV.pureindices
  )
  def cuttingplanefunction(self, x0, sigma, *args, **kwargs):
    if np.all(x0[self.gZ34indices,] == 0): #this happens for VBF when there are no reweighted ZZ fusion events in the bin
      return cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic_1stvariableonlyeven(x0, sigma, *args, **kwargs)

    elif np.all(x0[self.gZ3indices,] == 0): #this happens for VBF when there are no reweighted ZZ fusion events in the bin but there are events in the L1Zg sample
      return cuttingplanemethod4dquartic_4thvariablezerocubic_1stvariableonlyeven(x0, sigma, *args, **kwargs)

    elif max(
      ufloat(x0[i], sigma[i])
      for i in range(self.ntemplates)
      if i in self.gZ34indices and i in self.pureindices
    ) / min(
      ufloat(x0[i], sigma[i])
      for i in range(self.ntemplates)
      if i not in self.gZ34indices and i in self.pureindices
    ) < 1e-3:
      return cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_1stvariableonlyeven(x0, sigma, *args, **kwargs)

    else:
      return cuttingplanemethod4dquartic_1stvariableonlyeven(x0, sigma, *args, **kwargs)
  gZ3indices = tuple(i for i, _ in enumerate(templatenames) if "gl3" in _)
  gZ34indices = tuple(i for i, _ in enumerate(templatenames) if "gl3" in _ or _ == "l")
  cuttingplanehaspermutations = True

class FourParameterWWHVV_nog4int(ConstrainedTemplatesWithFit):
  templatenames = tuple(name for name in FourParameterWWHVV.templatenames if "gi1" not in name and "gi3" not in name)
  pureindices = tuple(
    index - sum(1 for i, name in enumerate(FourParameterWWHVV.templatenames) if i < index and ("gi1" in name or "gi3" in name))
    for index in FourParameterWWHVV.pureindices
  )
  cuttingplanefunction = staticmethod(cuttingplanemethod4dquartic_4thvariablequadratic_1stvariableonlyeven)
  cuttingplanehaspermutations = True

def findbincontentswrapper(self, args, **kwargs):  #args without * is correct
  try:
    return self.findbincontents(*args, **kwargs)
  except Exception as e:
    #https://stackoverflow.com/a/16618842/5228524
    raise Exception("".join(traceback.format_exception(*sys.exc_info())))
