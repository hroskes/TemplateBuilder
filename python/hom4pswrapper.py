import abc, contextlib, itertools, multiprocessing, os, re, socket, subprocess32 as subprocess

import numpy as np

nproc = multiprocessing.cpu_count()

def getcommandline(dispatch=None, engine=None, threads=None, pre=["balance", "const"], post=["pop", "pop", "refine", "summary"], mprec=None, summarydigits=None, homotopy="polyexp", stepctrl=False):
  args = ["hom4ps-core"]
  if dispatch is not None: args.append("--dispatch="+dispatch)
  if engine is not None: args.append("--engine="+engine)
  if mprec is not None:
    args.append("-m{:d}".format(mprec))
  if threads is not None:
    if threads.upper() == "CPU": threads = str(nproc)
    args.append("--threads="+threads)
  for _ in pre:
    args.append("--pre="+_)
  args += [
    "--sink=mem",
  ]
  if "homotopy" is not None:
    args.append("--homotopy="+homotopy)
  args += [
    "--monitor=facet-div"
  ]
  for _ in post:
    args.append("--post="+_)
  args += [
    "-fsummary-standard:1",
    "-fsummary-digits:{:d}".format(summarydigits),
  ]
  if stepctrl:
    args.append("-Mstep-ctrl")
  return args


def smallcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16)
def fastcmdline():
  return getcommandline(threads="CPU", summarydigits=16)
def smallparallelcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU")
def smallparalleltdegcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU", homotopy="tdeg")
def smallparalleltdegstepctrlcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU", homotopy="tdeg", stepctrl=True)
def smallparalleltdegnopostcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU", homotopy="tdeg", post=["summary"])
def easycmdline():
  return getcommandline(mprec=140, threads="CPU", pre=["bouncer", "balance", "const"], post=["pop", "pop", "refine-mp", "summary"], summarydigits=32)

def getcmdline(which):
  return {
    "small": smallcmdline,
    "fast": fastcmdline,
    "smallparallel": smallparallelcmdline,
    "smallparalleltdeg": smallparallelcmdline,
    "smallparalleltdegnopost": smallparalleltdegnopostcmdline,
    "smallparalleltdegstepctrl": smallparalleltdegstepctrlcmdline,
    "easy": easycmdline,
  }[which]()

class Hom4PSResult(object):
  superinitargs = ()
  def __init__(self, stdin, stdout, stderr):
    self.stdin = stdin
    self.stdout = stdout
    self.stderr = stderr
    super(Hom4PSResult, self).__init__(*self.superinitargs)
  @property
  def nfailedpaths(self):
    match = re.search(r"Failed Paths\s*:\s*([0-9]+)", self.stdout)
    try:
      return int(match.group(1))
    except AttributeError:
      print "-----------------------"
      print self.stdout
      print "-----------------------"
      raise
  @property
  def ndivergentpaths(self):
    match = re.search(r"Divergent:\s*([0-9]+)", self.stdout)
    try:
      return int(match.group(1))
    except AttributeError:
      print "-----------------------"
      print self.stdout
      print "-----------------------"
      raise
  @property
  def solutions(self):
    result = []
    for solution in self.stdout.split("\n\n"):
      if not re.search("solution *#", solution): continue
      match = re.search(r"values: *([^\n]*)", solution)
      if "This solution appears to be real" in solution:
        result.append(np.array([float(_) for _ in match.group(1).split()]))
      else:
        result.append(np.array([complex(re.sub(r"i[*]([0-9Ee.+-]+)", r"\1j", _).replace("+-", "-")) for _ in match.group(1).split()]))
    return result
  @property
  def realsolutions(self):
    return [solution for solution in self.solutions if np.all(np.isreal(solution))]
  @property
  def nduplicatesolutions(self):
    result = 0
    for solution1, solution2 in itertools.combinations(self.solutions, 2):
      if np.all(np.isclose(solution1, solution2, rtol=1e-7, atol=1e-10)):
        result += 1
    return result
  @property
  def variableorder(self):
    paragraphs = iter(self.stdout.split("\n\n"))
    for paragraph in paragraphs:
      if "The solution values are listed with the variables in the following order:" in paragraph: break
    return tuple(next(paragraphs).split())

class Hom4PSRuntimeError(Hom4PSResult, RuntimeError):
  @abc.abstractproperty
  def errormessage(self): pass
  @property
  def superinitargs(self): return self.errormessage+"\n\ninput:\n\n"+self.stdin+"\n\nstdout:\n\n"+self.stdout+"\n\nstderr:\n\n"+self.stderr,

class Hom4PSErrorMessage(Hom4PSRuntimeError):
  errormessage = "hom4ps printed an error message."

class Hom4PSFailedPathsError(Hom4PSRuntimeError):
  errormessage = "hom4ps found some failed paths."
  def __init__(self, *args, **kwargs):
    super(Hom4PSFailedPathsError, self).__init__(*args, **kwargs)
    assert self.nfailedpaths != 0

class Hom4PSDivergentPathsError(Hom4PSRuntimeError):
  errormessage = "hom4ps found some divergent paths."
  def __init__(self, *args, **kwargs):
    super(Hom4PSDivergentPathsError, self).__init__(*args, **kwargs)
    assert self.ndivergentpaths != 0

class Hom4PSDuplicateSolutionsError(Hom4PSRuntimeError):
  errormessage = "hom4ps found some duplicate solutions."
  def __init__(self, *args, **kwargs):
    super(Hom4PSDuplicateSolutionsError, self).__init__(*args, **kwargs)
    assert self.nduplicatesolutions != 0

class Hom4PSFailedAndDivergentPathsError(Hom4PSFailedPathsError, Hom4PSDivergentPathsError):
  errormessage = "hom4ps found some divergent paths and some failed paths."

class Hom4PSFailedPathsAndDuplicateSolutionsError(Hom4PSFailedPathsError, Hom4PSDuplicateSolutionsError):
  errormessage = "hom4ps found some failed paths and some duplicate solutions."

class Hom4PSDivergentPathsAndDuplicateSolutionsError(Hom4PSDivergentPathsError, Hom4PSDuplicateSolutionsError):
  errormessage = "hom4ps found some divergent paths and some duplicate solutions."

class Hom4PSFailedAndDivergentPathsAndDuplicateSolutionsError(Hom4PSFailedPathsError, Hom4PSDivergentPathsError, Hom4PSDuplicateSolutionsError):
  errormessage = "hom4ps found some failed paths, some divergent paths, and some duplicate solutions."

class Hom4PSTimeoutError(Hom4PSRuntimeError):
  errormessage = "hom4ps timed out."

@contextlib.contextmanager
def setenv(name, value):
  oldvalue = os.environ.get(name, None)
  os.environ[name] = value
  try:
    yield
  finally:
    if oldvalue is None:
      del os.environ[name]
    else:
      os.environ[name] = oldvalue

@contextlib.contextmanager
def addtocpath(path):
  oldvalue = os.environ.get("CPATH", None)
  if oldvalue is None:
    value = path
  else:
    value = oldvalue+":"+path
  with setenv("CPATH", value):
    yield

def runhom4ps(stdin, whichcmdline, verbose=False, timeout=10):
  if "bc-login" in socket.gethostname(): raise RuntimeError("Can't run hom4ps on MARCC login nodes")
  cmdline = getcmdline(whichcmdline)
  if verbose: print stdin

  with addtocpath(os.path.join(os.environ["CMSSW_BASE"], "include")):
    p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
      stdout, stderr = p.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired as e:
      raise Hom4PSTimeoutError(stdin, e.stdout if e.stdout else "", e.stderr if e.stderr else "")

  if "error" in stderr:
    raise Hom4PSErrorMessage(stdin, stdout, stderr)
  result = Hom4PSResult(stdin, stdout, stderr)

  if result.nfailedpaths or result.ndivergentpaths or result.nduplicatesolutions:
    if verbose: print stdout
    raise {
      (False, False, True ): Hom4PSDuplicateSolutionsError,
      (False, True,  False): Hom4PSDivergentPathsError,
      (False, True,  True ): Hom4PSDivergentPathsAndDuplicateSolutionsError,
      (True,  False, False): Hom4PSFailedPathsError,
      (True,  False, True ): Hom4PSFailedPathsAndDuplicateSolutionsError,
      (True,  True,  False): Hom4PSFailedAndDivergentPathsError,
      (True,  True,  True ): Hom4PSFailedAndDivergentPathsAndDuplicateSolutionsError,
    }[bool(result.nfailedpaths), bool(result.ndivergentpaths), bool(result.nduplicatesolutions)](stdin, stdout, stderr)
  if verbose: print stdout
  return result
