using MatrixDepot
using LinearAlgebra
using KLU
using SparseArrays
using Interpolations
using DataFrames
using Plots

# # Smart discovery

# Define wrappers to different LU algorithms and implementations
function umfpack_a(A)
    t = @elapsed  L, U, p = lu(A)
    err = norm(A[p,:] - L*U, 1)
    return t, err
end
function klu_a(A)
    sA = @views sparse(A)
    t = @elapsed K = klu(sA)
    err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
    return t, err
end
algs = [umfpack_a, klu_a]

# Define matrices
mat_names = ["baart", "cauchy",  "circul", "clement", "companion", 
             "deriv2", "dingdong", "fiedler", "forsythe", "foxgood","frank",
             "golub","gravity","grcar", "hankel","hilb","kahan","kms", "lehmer",
             "lotkin","magic","minij","moler","oscillate", "parter", "pei", 
             "prolate", "randcorr","rando","randsvd","rohess",
             "sampling","shaw", "spikes","toeplitz","tridiag","triw","ursell",
             "wilkinson","wing"]

# Define matrix sizes
ns = [10, 100, 1000]


# # Generate smart choice database trough smart discovery
df = DataFrame(mat_name = String[], n = Int[], algorithm = String[], 
               time = Float64[], error = Float64[])
for mat_name in mat_names
    for n in ns
        A = matrixdepot(mat_name, n)
        for a in algs
            try
                t, err = a(A)
                push!(df, [mat_name, n, "$(nameof(a))", t, err])
            catch e
                println("$(mat_name),$n, $(nameof(a)): an error occurred: $e")
            end
        end
        GC.gc()
    end
end

# # Show discovery process results
df

# TODO: Plot here

# # Generate smart choice model

#TODO: ML model here

smart_choice = Dict()
for mat_name in mat_names
    for n in ns
        df′ = @views df[(df.mat_name .== mat_name) .&& (df.n .== n), :]
        min_time = minimum(df′.time)
        min_time_row = df′[df′.time .== min_time, :]
        push!(smart_choice, (mat_name, n) => eval(Meta.parse(min_time_row.algorithm[1])))
    end
end

############

# All matrices
# mat_names =  ["baart","binomial","blur","cauchy","chebspec","chow","circul",
#              "clement","companion","deriv2","dingdong","erdrey","fiedler",
#              "forsythe","foxgood","frank","gilbert","golub","gravity","grcar",
#              "hadamard","hankel","heat","hilb","invhilb","invol","kahan","kms",
#              "lehmer","lotkin","magic","minij","moler","neumann","oscillate",
#              "parallax","parter","pascal","pei","phillips","poisson","prolate",
#              "randcorr","rando","randsvd","rohess","rosser","sampling","shaw",
#              "smallworld","spikes","toeplitz","tridiag","triw","ursell","vand",
#              "wathen","wilkinson","wing"]

# Problematic matrices: interface issues (e.g. dim^2 instead of dim), out of memory, singular value exception.
# mat_names = ["binomial", "blur", "poisson", "chow", "erdrey", "invol","neumann",
#              "parallax","pascal","rosser",,"vand", "smallworld","gilbert",
#              "hadamard","heat","invhilb", "wathen", "chebspec","phillips",] 

