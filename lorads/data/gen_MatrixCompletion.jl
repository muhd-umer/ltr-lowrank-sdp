using Pkg
Pkg.activate("./gen_data")
using Random
using SparseArrays
using LinearAlgebra
using Printf
using SparseMatricesCOO

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
            Am, An = An, Am
            A_rows = A.cols
            A_cols = A.rows
            A_vals = A.vals
        else
            quiet == 0 && println("A is not of the correct size to match b")
            return 1
        end
    else
        A_rows = A.rows
        A_cols = A.cols
        A_vals = A.vals
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
                    @printf(fid, "0 %d %d %d %.18e\n", i, II, JJ, -V)
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
            @printf(fid, "%d %d %d %d %.18e\n", cn, nsdpblocks + 1, c.nzind[k], c.nzind[k], -c.nzval[k])
        end
    end

    # Write constraint matrices
    for cn in 1:m
        base = sizelin + 1
        constr_idx_start = searchsortedfirst(A_rows, cn)
        constr_idx_end = searchsortedfirst(A_rows, cn + 1) - 1
        nzind = @view(A_cols[constr_idx_start:constr_idx_end])
        nzval = @view(A_vals[constr_idx_start:constr_idx_end])

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
                        @printf(fid, "%d %d %d %d %.18e\n", cn, i, II, JJ, V)
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
                @printf(fid, "%d %d %d %d %.18e\n", cn, nsdpblocks + 1, nzind[k], nzind[k], nzval[k])
            end
        end
    end

    close(fid)
    println("Done!")
    println("")
    return 0
end

"""
    sample_without_replacement(p, q, m)

Sample m unique indices from a p×q matrix without replacement.

Parameters:
- p: Number of rows
- q: Number of columns
- m: Number of samples to draw

Returns:
- Vector of sampled indices
"""
function sample_without_replacement(p, q, m)
    n = p * q
    if m > n
        throw(ArgumentError("m cannot be larger than p*q"))
    end

    sampled_indices = Dict{Int, Int}()
    sample = Vector{Int}(undef, m)
    
    for i in 1:m
        while true
            candidate = rand(1:n)
            if !haskey(sampled_indices, candidate)
                sampled_indices[candidate] = i
                sample[i] = candidate
                break
            end
        end
    end

    return sample
end

"""
    gen_mc(size, rank, sampleRatio)

Generate a matrix completion problem instance.

This implementation follows the matrix completion problem formulation and generation method
from ManiSDP [1].

Parameters:
- size: Size of the PSD matrix of the corresponding SDP problem
- rank: Rank of the underlying matrix
- sampleRatio: sample ratio (number of observed entries = sampleRatio * size)

Returns:
- At: Constraint matrix
- b: Right-hand side vector
- c: Objective function vector
- K: Cone structure dictionary

References:
[1] Jie Wang and Liangbing Hu, "Solving Low-Rank Semidefinite Programs via Manifold Optimization", 2023.
    Available at: https://github.com/wangjie212/ManiSDP-matlab
"""
function gen_mc(size, rank, spRatio)
    Random.seed!(100)      

    p = round(Int, size / 2)
    q = round(Int, size / 2)
    n = p + q
    m0 = spRatio * n
    # draw, dedupe, sort
    Omega = sort(unique(rand(1:p*q, m0)))
    m = length(Omega)

    # build the objective: c = vec(I_n)
    indices = collect((0:n-1) * n .+ (1:n))
    values = ones(n)
    @time c = SparseVector(n*n, indices, values)
    # precompute factors 
    A = randn(p, rank)      # p×r
    B = randn(rank, q)      # r×q

    b = zeros(m)
    row = Vector{Int}(undef, 2*m)
    col = Vector{Int}(undef, 2*m)
    val = fill(1.0, 2*m)

    for i in 1:m
        idx = Omega[i]
        j = ceil(Int, idx / q)
        k = idx % q == 0 ? q : idx % q

        # M(j,k) = dot(A[j,:], B[:,k])
        b[i] = 2 * dot(@view(A[j, :]), @view(B[:, k]))

        row[2*i-1] = (j-1)*n + (k + p)    # X[k+p, j]
        row[2*i]   = (k + p - 1)*n + j    # X[j, k+p]
        col[2*i-1] = i
        col[2*i]   = i
    end

    At = SparseMatrixCOO(n^2, m, row, col, val)
    K = Dict(:l => 0, :s => [n])
    return At, b, c, K
end

# Example usage
for size in [500, 1000]
    @time At, b, c, K = gen_mc(size, 10, 400)
    @time writesdpa("Matrix_Completion_SDP/MC_$(size).dat-s", At, b, c, K)
end

