print("Imports successful")
try:
    import scipy.io

    print("scipy.io available")
except ImportError:
    print("scipy.io missing")
try:
    import pyxdf

    print("pyxdf available")
except ImportError:
    print("pyxdf missing")
