function permcp(tensorname,method,minrank,maxrank,nreps,shiftAmount)
    %PERMCP Computes a permuted CP decomposition for several choices
    % for the number of components (minrank up to maxrank) and a given
    % number of repetitions using different random initializations.
    % It assumes each fiber in the tensor is a flattened 2D array,
    % with number of rows given by `shiftAmount`. These will be permuted
    % by circular-shifting the rows to make the resulting factors
    % agnostic to each neuron's preferred direction of motion.
    % Is saves the output for each # components as a .mat file containing 
    % the resulting factors, objective, fit, and lambdas for each repetition.

    % This function calls modified versions of files from Tensor Toolbox:
    % Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.1,
    % www.tensortoolbox.org, June 2019. https://gitlab.com/tensors/tensor_toolbox.

    % using the direct optimization (OPT) method from
    % E. Acar, D. M. Dunlavy and T. G. Kolda. A Scalable Optimization Approach for Fitting 
    % Canonical Tensor Decompositions, Journal of Chemometrics 25(2):67-86, February 2011. 
    % (http://dx.doi.org/10.1002/cem.1335).

 
    addpath('tensor_toolbox');
    addpath('L-BFGS-B-C/Matlab');

    X = load([tensorname '.mat']);
    X = double(X.X);
    assert(min(X(:)) >= 0);
    tensorX = tensor(X);
    normX = norm(tensorX);


    cp_opt_options = struct('printEvery',0,'factr',1e-5,'pgtol',1e-4,'maxIts',1000);

    for RANK=minrank:1:maxrank
        fprintf('RANK %d:\n',RANK);
        bestFit = inf;

        results = [];

        for rep=1:nreps
    	fprintf('\n%d:',rep);
            switch method
                case 'shift'
                    [M1,~,out] = my_cp_opt(tensorX,RANK,'init','rand','lower',0,'opt_options',cp_opt_options,'shift',shiftAmount);
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
            res.fit = norm(tensorX-fullM1)/normX;
            res.fullM = fullM1;


            res.P = M1;
            results = [results res];
            fprintf(',');
        end
        fprintf('\n');

        % sort reps by objective (rec. error)
        results = SortArrayofStruct( results, 'obj', 'ascend' );


        factors = {};
        fits = {};
        objs = {};
        lams = {};

        for r=1:length(results)
            res = results(r);
            P = res.P;
            lams{r} = P.lambda';%'

            fits{r} = res.fit;
            objs{r} = res.obj;
            factors{r} = {};

            for dim=1:length(P.u)
                factors{r}{dim} = P.u{dim};
            end
        end

        save([tensorname '_rank' num2str(RANK,'%02d') '.mat'],'lams','factors','fits','objs');
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
