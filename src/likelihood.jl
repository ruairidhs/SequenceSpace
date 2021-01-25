
function fastcov(input, cache)

    # Computes autocovariances using the fast fourier transform
    # Input is a three dimensional array of MA coefficients:
    #   - Input[o, z, s] is the coefficient at lag s for output o in response to shock z
    # cache is pre-allocated complex array no * no * (ns + 1)

    dft = rfft(cat(input, zeros(axes(input)), dims = 3), 3)
    for i in 1:size(dft, 3)
        mul!(view(cache, :, :, i), conj.(dft[:, :, i]), transpose(dft[:, :, i]))
    end
    return irfft(cache, size(input, 3) * 2, 3)[:, :, axes(input, 3)]
end

function makefftcache(input)
    # Same input as for fastcov, makes the cache
    zeros(Complex{Float64}, size(input, 1), size(input, 1), size(input, 3) + 1)
end

function makeinput(no, nz, T)
    # initializes the array containing the MA coefficients
    # no is number of obs vars, nz is number of shocks and T as in MA(T-1)
    zeros(no, nz, T)
end

function updateMAcoefficients!(input, obsvars, shockmat, Gs)

    # Computes the MA coefficients that can be passed to fastcov
    # shockmat is T × nz matrix giving the MA coefficients of each exogenous variable

    for oi in eachindex(obsvars)
        for zi in axes(shockmat, 2)
            mul!(view(input, oi, zi, :), Gs[obsvars[oi]], shockmat[:, zi])
        end
    end
    return input
end

function makeV(autocovs, Tobs)

    # returns cholesky decomposition of V (p. 34). autocovs is output of fastcov

    no = size(autocovs, 1)
    V = zeros(no * Tobs, no * Tobs)

    # Fill in V using each panel of autocovs
    # Only need to fill half of the matrix as it is symmetric
    # WARNING! currently only works for Tobs < T, need to add zeros to generalize
    rloc = 1
    for t1 in 1:Tobs
        cloc = rloc # start on the diagonal
        for t2 in 1:Tobs-t1+1 # don't need to go through all Tobs on later rows
            V[rloc:rloc+no-1, cloc:cloc+no-1] .= autocovs[:, :, t2] # need to change to .= for multidimensional
            cloc += no
        end
        rloc += no
    end

    return cholesky!(Symmetric(V))

end

function _likelihood(V, obsdata)
    # V is cholesky matrix, obsdata is stacked vector of observations
    -0.5*(logdet(V) + dot(obsdata, inv(V), obsdata))
end