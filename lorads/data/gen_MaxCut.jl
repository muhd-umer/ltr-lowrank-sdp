using Pkg
Pkg.activate("./gen_data")
using LinearAlgebra
using SparseArrays
using MAT
using SparseMatricesCOO
using Printf
using FileIO
using SplitApplyCombine

"""
    writesdpa(fname::String, A, b, c, K, pars=nothing)

Write SDPA format file for semidefinite programming problems.

Parameters:
- fname: Output file name
- A: Constraint matrix
- b: Right-hand side vector
- c: Objective function vector
- K: Cone structure dictionary
- pars: Optional parameters dictionary for controlling output behavior

Returns:
- 0 on success, 1 on failure
"""
function writesdpa(fname::String, A, b, c, K, pars=nothing)
    println("Writing into $fname ...")
    println("")
    
    # Initialize parameters
    quiet = 0
    check = 0
    if !isnothing(pars)
        quiet = get(pars, :printlevel, 1) == 0 ? 1 : 0
        check = get(pars, :check, 0)
    else
        pars = Dict(:printlevel => 1, :check => 0)
    end

    # Validate input data
    if !isreal(A) || !isreal(b) || !isreal(c)
        quiet == 0 && println("Input data contains complex numbers!")
        return 1
    end

    # Validate cone constraints
    for (cone_type, msg) in [
        (:q, "quadratic cone constraints"),
        (:r, "rotated cone constraints"),
        (:f, "free variables")
    ]
        if haskey(K, cone_type) && !isempty(K[cone_type]) && K[cone_type] != 0
            quiet == 0 && println("$msg are not supported.")
            return 1
        end
    end

    # Get problem dimensions
    m = length(b)
    Am, An = size(A)

    # Handle matrix transpose if needed
    if Am != m
        if An == m
            quiet == 0 && println("Transposing A to match b")
            A = transpose(A)
            Am, An = An, Am
        else
            quiet == 0 && println("A is not of the correct size to match b")
            return 1
        end
    end

    # Handle zero objective function
    if c == 0 || isempty(c)
        quiet == 0 && println("Expanding c to appropriate size")
        c = spzeros(An, 1)
    end

    # Get block structure information
    nlin = get(K, :l, 0)
    sizelin = isempty(nlin) || nlin == 0 ? 0 : nlin

    nsdpblocks = 0
    sizesdp = 0
    if haskey(K, :s)
        nsdpblocks = length(K[:s])
        sizesdp = sum(K[:s] .^ 2)
        if isempty(sizesdp) || K[:s] == 0
            nsdpblocks = 0
            sizesdp = 0
        end
    end

    nblocks = nsdpblocks + (nlin > 0 ? 1 : 0)

    # Print problem statistics
    if quiet == 0
        println("===== Problem statistics =====")
        println("==> Number of constraints: $m")
        println("==> Number of SDP blocks: $nsdpblocks")
        println("==> Dims of SDP blocks: $(K[:s])")
        println("==> Number of LP vars: $nlin")
        println("")
    end

    # Write SDPA file
    fid = open(fname, "w")
    
    # Write header information
    println(fid, "$m")
    println(fid, "$nblocks")
    
    # Write block structure
    if K[:s][1] > 0
        print(fid, join(K[:s], " "))
    end
    if nlin > 0
        print(fid, " $(nlin * -1)")
    end
    println(fid)
    
    # Write right-hand side vector
    println(fid, join(b, " "))

    # Write objective function matrix
    base = sizelin + 1
    for i in 1:nsdpblocks
        idx_start = searchsortedfirst(c.nzind, base)
        idx_end = searchsortedlast(c.nzind, base + K[:s][i]^2)
        
        cnt = idx_end - idx_start + 1
        if cnt != 0
            for k in 1:cnt
                II = (c.nzind[idx_start + (k-1)] - base) % K[:s][i] + 1
                JJ = (c.nzind[idx_start + (k-1)] - base) ÷ K[:s][i] + 1
                V = c.nzval[idx_start + (k-1)]
                if II <= JJ && V != 0
                    println(fid, "0 $i $(II) $(JJ) $(-V)")
                end
            end
        end
        base += K[:s][i]^2
    end

    # Write linear block of objective function
    idx_start = searchsortedfirst(c.nzind, sizelin + 1)
    nnzlin = idx_start - 1
    if nnzlin > 0
        for k in 1:nnzlin
            println(fid, "$cn $(nsdpblocks + 1) $(c.nzind[k]) $(c.nzind[k]) $(-c.nzval[k])")
        end
    end

    # Write constraint matrices
    for cn in 1:m
        base = sizelin + 1
        constr_idx_start = searchsortedfirst(A.rows, cn)
        constr_idx_end = searchsortedfirst(A.rows, cn + 1) - 1
        nzind = @view(A.cols[constr_idx_start:constr_idx_end])
        nzval = @view(A.vals[constr_idx_start:constr_idx_end])

        for i in 1:nsdpblocks
            idx_start = searchsortedfirst(nzind, base)
            idx_end = searchsortedlast(nzind, base + K[:s][i]^2)
            
            cnt = idx_end - idx_start + 1
            if cnt != 0
                for k in 1:cnt
                    II = (nzind[idx_start + (k-1)] - base) % K[:s][i] + 1
                    JJ = (nzind[idx_start + (k-1)] - base) ÷ K[:s][i] + 1
                    V = nzval[idx_start + (k-1)]
                    if II <= JJ
                        println(fid, "$cn $i $(II) $(JJ) $(V)")
                    end
                end
            end
            base += K[:s][i]^2
        end

        # Write linear part of constraints
        idx_start = searchsortedfirst(nzind, sizelin + 1)
        nnzlin = idx_start - 1
        if nnzlin > 0
            for k in 1:nnzlin
                println(fid, "$cn $(nsdpblocks + 1) $(nzind[k]) $(nzind[k]) $(nzval[k])")
            end
        end
    end

    close(fid)
    println("Done!")
    println("")
    return 0
end

"""
    create_sedumi_maxcut(prob_name, prob_path="./")

Create SDP formulation for MaxCut problem.

Parameters:
- prob_name: Name of the problem
- prob_path: Path to the problem data file

Returns:
- A: Constraint matrix
- b: Right-hand side vector
- c: Objective function vector
- K: Cone structure dictionary
"""
function create_sedumi_maxcut(prob_name, prob_path="./")
    # Read adjacency matrix
    mat_file_name = joinpath(prob_path, prob_name * ".mat")
    mat_data = matread(mat_file_name)

    A = mat_data["Problem"]["A"]
    
    # Compute weighted degree matrix
    row_sums = sum(A, dims=2)
    n = length(row_sums)
    D = sparse(1:n, 1:n, row_sums[:, 1], n, n)
    
    # Compute Laplacian matrix
    L = D .- A
    
    # Scale Laplacian matrix by 1/4 for MaxCut formulation
    c = -0.5 .* sparsevec(L[:])

    dim = size(A, 1)

    # Construct constraint matrix A for SeDuMi
    I = collect(1:dim)
    J = (I .- 1) .* dim .+ I
    S = ones(dim)
    Aeq = SparseMatrixCOO(dim, dim^2, I, J, S)

    beq = ones(dim)
    K = Dict(:l => 0, :s => [size(Aeq, 1)])

    return Aeq, beq, c, K
end

"""
    batch_process_maxcut(mat_folder::String, output_folder::String="./")

Process all .mat files in a folder and generate corresponding .dat-s files.

Parameters:
- mat_folder: Path to the folder containing .mat files
- output_folder: Path to the output folder for .dat-s files, defaults to current directory

Returns:
- Number of files processed
"""
function batch_process_maxcut(mat_folder::String, output_folder::String="./")
    # Ensure output folder exists
    mkpath(output_folder)
    
    # Get all .mat files
    mat_files = filter(x -> endswith(x, ".mat"), readdir(mat_folder))
    
    # Process each file
    for mat_file in mat_files
        # Get filename without extension
        prob_name = splitext(mat_file)[1]
        
        println("Processing file: $prob_name")
        
        # Generate SDP problem
        @time A, b, c, K = create_sedumi_maxcut(prob_name, mat_folder)
        
        # Generate output file path
        output_file = joinpath(output_folder, "$(prob_name).dat-s")
        
        # Write .dat-s file
        @time writesdpa(output_file, A, b, c, K)
        
        println("Completed file: $prob_name")
        println("------------------------")
    end
    
    return length(mat_files)
end

# Example usage
batch_process_maxcut("Max_cut_initial_data", "Max_cut_initial_data")