__precompile__()

module RegisterUtilities

import Base: done, next, start
export Counter

#### Counter ####
#
# Stolen from Grid.jl. Useful when you want to do more math on the iterator.

immutable Counter
    max::Vector{Int}
end
Counter(sz::Tuple) = Counter(Int[sz...])

function start(c::Counter)
    N = length(c.max)
    state = ones(Int,N)
    if N > 0
        state[1] = 0 # because of start/done/next sequence, start out one behind
    end
    return state
end
function done(c::Counter, state)
    if isempty(state)
        return true
    end
    # we do the increment as part of "done" to make exit-testing more efficient
    state[1] += 1
    i = 1
    max = c.max
    while state[i] > max[i] && i < length(state)
        state[i] = 1
        i += 1
        state[i] += 1
    end
    state[end] > max[end]
end
next(c::Counter, state) = state, state

end
