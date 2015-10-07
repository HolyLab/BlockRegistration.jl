function quadratic(m, n, shift, Q)
    A = zeros(m, n)
    c = block_center(m, n)
    cntr = [shift[1]+c[1], shift[2]+c[2]]
    u = zeros(2)
    for j = 1:n, i = 1:m
        u[1], u[2] = i-cntr[1], j-cntr[2]
        A[i,j] = dot(u,Q*u)
    end
    A
end

function block_center(sz...)
    ntuple(i->sz[i]>>1+1, length(sz))
end