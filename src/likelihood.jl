
function fastcov(input, cache, T)

    # Computes autocovariances using the fast fourier transform
    # Input is a three dimensional array of MA coefficients:
    #   - Input[o, z, s] is the coefficient at lag s for output o in response to shock z
    # cache is pre-allocated complex array no * no * (ns + 1)
    #dft = rfft(cat(input, zeros(axes(input)), dims = 3), 3)
    dft = rfft(input, 3)
    for i in 1:size(dft, 3)
        mul!(view(cache, :, :, i), conj.(dft[:, :, i]), transpose(dft[:, :, i]))
    end
    return irfft(cache, size(input, 3), 3)[:, :, 1:T]
end

function fastcov(input, cache, P, Pinv, T)
    # P is a precomputed FFT plan
    dft = P * input
    for i in 1:size(dft, 3)
        mul!(view(cache, :, :, i), conj.(dft[:, :, i]), transpose(dft[:, :, i]))
    end
    return (Pinv * cache)[:, :, 1:T]
    # mul!(autocovs, Pinv, cache)
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
    T = size(shockmat, 1)
    for oi in eachindex(obsvars)
        loc = 1
        for zi in axes(shockmat, 2)
            mul!(view(input, oi, zi, 1:T), view(Gs[obsvars[oi]], :, loc:loc+T-1), shockmat[:, zi])
            loc += T
        end
    end
    return input
end

function getcorrelations(var1, vars, shockcoefs, Gs)

    # Given a matrix of shock coefficients,
    # computes Corr(dX_(t+l), dYt) for Y=var1 and X=[var1, vars]

    T = size(shockcoefs, 1)
    vs = pushfirst!(copy(vars), var1)
    input = makeinput(length(vs), size(shockcoefs, 2), T)
    padded_input = cat(input, zeros(axes(input)), dims=3)
    updateMAcoefficients!(padded_input, vs, shockcoefs, Gs)
    fft_cache = makefftcache(input)

    autocovs = fastcov(padded_input, fft_cache, T)

    sds = sqrt.(diag(autocovs[:, :, 1]))

    forward_corrs  = [autocovs[1, :, t] ./ (sds * sds[1]) for t in 1:T]
    backward_corrs = [autocovs[:, 1, t] ./ (sds * sds[1]) for t in T:-1:2]
    ordered_corrs  = vcat(permutedims.(vcat(backward_corrs, forward_corrs))...)
    return ordered_corrs
end

function updateV!(V, autocovs, Tobs)

    # returns cholesky decomposition of V (p. 34). autocovs is output of fastcov

    no, T2 = size(autocovs, 1), size(autocovs, 3)
    # V = zeros(no * Tobs, no * Tobs)

    # Fill in V using each panel of autocovs
    # Only need to fill half of the matrix as it is symmetric
    # WARNING! currently only works for Tobs < T, need to add zeros to generalize
    
    for tcol in 1:Tobs
        c = no*tcol
        for trow in 1:tcol
            r = no*trow
            @views V[r-no+1:r, c-no+1:c] .= autocovs[:, :, tcol-trow+1]
        end
    end

    return cholesky!(Symmetric(V))
end

function _likelihood(V, obsdata)
    # V is cholesky matrix, obsdata is stacked vector of observations
    # obsdata should be stacked on second dimension first!
    -0.5*(logdet(V) + dot(obsdata, V \ obsdata))
end

function make_likelihood(shockfunc, nz, obsdata, obsvars, Gs)

    T = size(first(Gs)[2], 1)
    Tobs, no = size(obsdata) # number of observed time series, one for each shock
    # nz is number of shocks

    datavector = vec(permutedims(obsdata))

    input_array = makeinput(no, nz, T)
    padded_input = cat(input_array, zeros(axes(input_array)), dims=3)

    fft_cache = makefftcache(input_array)
    P    = plan_rfft(padded_input, 3) # 3 means perform the FFT along 3rd dim, i.e. T
    Pinv = plan_irfft(fft_cache, size(padded_input, 3), 3)
    V = zeros(no * Tobs, no * Tobs)
    shockmat = ones(T, nz)

    function l(Ω)
        shockfunc(shockmat, Ω)
        updateMAcoefficients!(padded_input, obsvars, shockmat, Gs)
        return _likelihood(
            updateV!(V, fastcov(padded_input, fft_cache, P, Pinv, T), Tobs), datavector
        )
        #return _likelihood(makeV(fastcov(padded_input, fft_cache, P, Pinv, T), Tobs), datavector)
    end
    
end