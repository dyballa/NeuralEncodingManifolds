The permuted tensor CP decomposition is performed using Tensor Toolbox, for MATLAB:

Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB.
www.tensortoolbox.org
https://gitlab.com/tensors/tensor_toolbox

To run it, first place the following files into your tensor_toolbox directory:

perm_cp_opt.m
perm_tt_cp_fg.m
perm_tt_cp_fun.m


Then call the function run_permcp by passing the necessary arguments (see documentation in  the `run_permcp.m` file for the details).

Each choice for the number of components, F, will generate an output file containing the resultant components, their corresponding lambdas, and the objective value for each different initialization.