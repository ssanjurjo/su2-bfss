# Lowest eigenpairs of the Hermitian pencil (H, M) by block shift-invert subspace iteration with a CHOLMOD Cholesky factor of
# H - sigma M (positive definite for sigma below the spectrum).  Input/output: npz files written/read by solve_ch.py.
# Usage: julia --project=<this folder> eig_chol.jl in.npz out.npz   (called by solve_ch.py; JL_BLAS_THREADS sets the BLAS threads)
using NPZ, SparseArrays, LinearAlgebra, Random
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "JL_BLAS_THREADS", string(Sys.CPU_THREADS))))
d = npzread(ARGS[1]); n = Int(d["n"]); nev = Int(d["nev"]); σ = Float64(d["sigma"])
H = SparseMatrixCSC(n, n, Int64.(d["h_indptr"]) .+ 1, Int64.(d["h_indices"]) .+ 1, complex.(d["hre"], d["him"]))
M = SparseMatrixCSC(n, n, Int64.(d["m_indptr"]) .+ 1, Int64.(d["m_indices"]) .+ 1, complex.(d["mre"], d["mim"]))
upper = haskey(d, "upper") && Int(d["upper"]) == 1
d = nothing; GC.gc()
t0 = time()
# with upper-triangle storage (upper = 1) the full operators are U + U' - Diag(U)
Hd = upper ? real.(diag(H)) : nothing; Md = upper ? real.(diag(M)) : nothing
mulH(X) = upper ? H * X + H' * X - Hd .* X : H * X
mulM(X) = upper ? M * X + M' * X - Md .* X : M * X
F = cholesky(Hermitian(H - σ * M, :U)); GC.gc()
# nnz(F) and the supernodal storage xsize are read from the factor itself; converting it to a sparse matrix would copy it twice
println("CHOLMOD factor: nnz(L) = $(nnz(F)), supernodal storage $(unsafe_load(pointer(F)).xsize) entries ($(round(time() - t0, digits=1)) s)"); flush(stdout)
p = nev + 6; Random.seed!(1); X = randn(ComplexF64, n, p)
λ = zeros(p); res = fill(Inf, nev)
MAXIT = parse(Int, get(ENV, "JL_MAXIT", "80")); TOL1 = parse(Float64, get(ENV, "JL_TOL1", "1e-10"))
for it in 1:MAXIT
    global X, λ, res
    Y = F \ mulM(X)
    Q = Matrix(qr(Y).Q)                                   # orthonormal basis of the new subspace
    Hs = Q' * mulH(Q); Ms = Q' * mulM(Q)
    e = eigen(Hermitian((Hs + Hs') / 2), Hermitian((Ms + Ms') / 2)); λ = real.(e.values); X = Q * e.vectors
    HX = mulH(X[:, 1:nev]); MX = mulM(X[:, 1:nev]); R = HX - MX * Diagonal(λ[1:nev])
    res = [norm(R[:, i]) / (abs(λ[i]) * norm(MX[:, i]) + norm(HX[:, i])) for i in 1:nev]
    (it % 5 == 0 || it < 3) && (println("iteration $it: lowest $(round.(λ[1:nev], digits=6)), rel. residuals $(round.(res, sigdigits=2))"); flush(stdout))
    (res[1] < TOL1 && maximum(res) < 1e-6) && (println("converged at iteration $it: residuals $(round.(res, sigdigits=2))"); break)
    it == MAXIT && println("converged? stopped at iteration $it (limit): lowest $(round.(λ[1:nev], digits=7)), residuals $(round.(res, sigdigits=2))")
end
npzwrite(ARGS[2], Dict("ev" => λ[1:nev], "xre" => real.(X[:, 1:nev]), "xim" => imag.(X[:, 1:nev]), "res" => res))
