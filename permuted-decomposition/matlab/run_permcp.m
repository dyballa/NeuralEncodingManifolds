function run_permcp(tensorname,method,minF,maxF,nreps,shiftAmount)
    %PERMCP Computes a permuted CP decomposition for several choices
    % of the number of components (minF up to maxF) and a given
    % number of repetitions using different random initializations.
    % It assumes each fiber in the tensor is a flattened 2D array,
    % with number of rows given by `shiftAmount`. These will be permuted
    % by circular-shifting the rows to make the resulting factors
    % agnostic to each neuron's preferred direction of motion.
    % Is saves the output for each F as a .mat file containing 
    % the resulting factors, lambdas, and objective for each repetition.

    % This function calls modified versions of files from Tensor Toolbox:
    % Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.1,
    % www.tensortoolbox.org, June 2019. https://gitlab.com/tensors/tensor_toolbox.

    % using the direct optimization (OPT) method from:
    % E. Acar, D. M. Dunlavy and T. G. Kolda. A Scalable Optimization Approach for Fitting 
    % Canonical Tensor Decompositions, Journal of Chemometrics 25(2):67-86, February 2011. 
    % (http://dx.doi.org/10.1002/cem.1335).

 
    % NOTE: you may need to modify these following two paths depending on where this file 
    % is located relative to  your installation of Tensor Toolbox:
    addpath('tensor_toolbox'); 
    addpath('L-BFGS-B-C/Matlab');

    X = load([tensorname '.mat']);
    X = double(X.X);
    assert(min(X(:)) >= 0);
    tensorX = tensor(X);
    normX = norm(tensorX);


    cp_opt_options = struct('printEvery',0,'factr',1e-5,'pgtol',1e-4,'maxIts',1000);

    for F=minF:1:maxF
        fprintf('F %d:\n',F);
        bestFit = inf;

        results = [];

        for rep=1:nreps
    	fprintf('\n%d:',rep);
            switch method
                case 'shift'
                    [M1,~,out] = perm_cp_opt(tensorX,F,'init','rand','lower',0,'opt_options',cp_opt_options,'shift',shiftAmount);
                otherwise
                    fprintf('ERROR: Method not recognized: %s\n', method);
                    return;
            end
                if isempty(strfind(out.ExitMsg,'CONVERGENCE'))
                    fprintf('rep %d: %s\n',rep,out.ExitMsg);
                end
                objf = 100 - out.Fit;

            if objf < bestFit
                bestFit = objf;
                fprintf('%.3f',bestFit);
            end
            
            res.obj = objf;
            fullM1 = full(M1);
            res.fullM = fullM1;


            res.P = M1;
            results = [results res];
            fprintf(',');
        end
        fprintf('\n');

        % sort reps by objective (rec. error)
        results = SortArrayofStruct( results, 'obj', 'ascend' );


        factors = {};
        objs = {};
        lams = {};

        for r=1:length(results)
            res = results(r);
            P = res.P;
            lams{r} = P.lambda';%'

            objs{r} = res.obj;
            factors{r} = {};

            for dim=1:length(P.u)
                factors{r}{dim} = P.u{dim};
            end
        end

        save([tensorname '_F' num2str(F,'%02d') '_nreps' num2str(nreps) '.mat'],'lams','factors','objs');
    end
end



function outStructArray = SortArrayofStruct( structArray, fieldName, direction)
    if ( ~isempty(structArray) &&  length(structArray)>0 )
      [~,I] = sort(arrayfun (@(x) x.(fieldName), structArray), direction) ;
      outStructArray = structArray(I) ;        
    else 
        disp ('Array of struct is empty');
    end      
end
