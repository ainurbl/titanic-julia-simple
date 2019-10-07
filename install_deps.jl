#=
install_deps:
- Julia version: 
- Author: bravit
- Date: 2019-10-07
=#

using Pkg

Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("ScikitLearn")
Pkg.add("XGBoost")
Pkg.add("Conda")

ENV["PYTHON"]=""
Pkg.build("PyCall")

using Conda
Conda.add("scikit-learn")
