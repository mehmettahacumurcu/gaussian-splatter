# Pins `colab` as a regular package so `from colab.verify_helpers import ...`
# (notebooks run with sys.path.insert(0, ".")) can never be shadowed by a
# same-named namespace portion elsewhere on sys.path.
