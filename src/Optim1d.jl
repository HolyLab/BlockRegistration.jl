module Optim1d

export bracket

# Bracket a function
function bracket{T}(f::Function, a::T, b::T, fa = f(a), fb = f(b); limit = typemax(T), iterations = 1000)
    const GOLD = convert(T, 1.618)
    if !isfinite(fa) || !isfinite(fb)
        error("Function value must be finite at two candidate points")
    end
    if fb > fa
        c = (b+GOLD*a)/(1+GOLD)
        fc = f(c)
        iter = 1
        while fc > fa && abs(b-a) > eps(T)*(abs(a)+abs(b)) && iter < iterations
            b, fb = c, fc
            c = (b+GOLD*a)/(1+GOLD)
            fc = f(c)
            iter += 1
        end
        if iter >= iterations
            error("Minimum not bracketed within $iterations iterations")
        end
        return a, c, b
    end
    c = b + GOLD*(b-a)
    fc = f(c)
    iter = 1
    while !isfinite(fc) && iter < iterations
        limit = min(limit, c)
        c = (b+limit)/2
        fc = f(c)
        iter += 1
    end
    while fb > fc && iter < iterations
        a, fa = b, fb
        b, fb = c, fc
        c = min(b + GOLD*(b-a), (b+limit)/2)
        fc = f(c)
        iter += 1
        while !isfinite(fc) && iter < iterations
            limit = min(limit, c)
            c = (b+limit)/2
            fc = f(c)
            iter += 1
        end
    end
    if iter >= iterations
        error("Minimum not bracketed within $iterations iterations")
    end
    a, b, c
end

end
