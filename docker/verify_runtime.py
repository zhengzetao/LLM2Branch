"""Dependency-only smoke check; uses no project data or external services."""
import json
import platform
import sys
import numpy
import scipy
import requests
import pyscipopt

model = pyscipopt.Model()
model.hideOutput()
for method in ("getState", "getKhalilState", "executeBranchRule"):
    if not callable(getattr(model, method, None)):
        raise RuntimeError("Missing custom solver API: " + method)
model.setIntParam("branching/vanillafullstrong/priority", 0)
x = model.addVar("x", vtype="B", obj=-3)
y = model.addVar("y", vtype="B", obj=-2)
model.addCons(2*x + 2*y <= 3)
model.optimize()
assert str(model.getStatus()) == "optimal", model.getStatus()
assert abs(model.getObjVal() + 3) < 1e-8
print(json.dumps({
    "platform": platform.machine(), "python": sys.version.split()[0],
    "numpy": numpy.__version__, "scipy": scipy.__version__,
    "requests": requests.__version__, "pyscipopt": pyscipopt.__version__,
    "solver_smoke": "optimal", "custom_apis": "available",
}, sort_keys=True))
