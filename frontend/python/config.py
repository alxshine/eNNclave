import os
import sys

_environment_variable = 'ENNCLAVE_HOME'
_ennclave_home = os.getenv(_environment_variable, default=None)


def get_ennclave_home():
    global _environment_variable
    global _ennclave_home
    if _ennclave_home is None:
        print(f"WARNING: environment variable {_environment_variable} not set, defaulting to current working directory",
              file=sys.stderr)
        _ennclave_home = os.getcwd()
    return _ennclave_home
