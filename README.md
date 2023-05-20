# BulkBoundaryBounds

Analysis code for

Hasegawa, Y. Unifying speed limit, thermodynamic uncertainty relation and Heisenberg principle via bulk-boundary correspondence. Nature Communications 14, 2828 (2023). 
https://doi.org/10.1038/s41467-023-38074-8

Author:

* Yoshihiko Hasegawa: hasegawa@biom.t.u-tokyo.ac.jp

Department of Information and Communication Engineering,
Graduate School of Information Science and Technology,
The University of Tokyo

## Code
- SpeedLimitClassical.ipynb
  - Calculation for classical speed limit
- SpeedLimitQuantum.ipynb
  - Calculation for quantum speed limit
- batch_ClassicalTUR.py
  - Run simulation for classical TUR. 
  - To execute, type `python batch_ClassicalTUR.py`
  - To record output, type `python batch_ClassicalTUR.py > output.txt`
- batch_QuantumTUR.py
  - Run simulation for quantum TUR.
  - To execute, type `python batch_QuantumTUR.py`
  - To record output, type `python batch_QuantumTUR.py > output.txt`

## Dependence
- Python
  - numpy
  - scipy
  - sympy
  - ujson
  - networkx
  - matplotlib
  - pandas
- Julia
   - StatsBase
   - PyCall
