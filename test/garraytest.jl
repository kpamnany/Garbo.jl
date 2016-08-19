#!/usr/bin/env julia

using Garbo

immutable Aelem
    num::Int64
    num2::Int64
end

@inline nputs(nid, s) = ccall(:puts, Cint, (Ptr{Int8},), string("[$nid] ", s))
@inline ptest(r, s) = nputs(nodeid, string(r ? "passed: " : "failed: ", s))

macro tst(ex)
    oex = Expr(:inert, ex)
    quote
        r = $ex
        nputs(nodeid, string((r ? "passed: " : "failed: "), $oex))
    end
end

if nodeid == 1
    ccall(:puts, Cint, (Ptr{Int8},), string("garraytest -- $nnodes nodes\n"))
end

# even distribution
# ---
nelems = nnodes * 5

# create the array
ga = Garray(Aelem, nelems)
@tst ndims(ga) == 1
@tst length(ga) == nnodes * 5
@tst size(ga) == tuple(nnodes * 5)

# get the local part
lo, hi = distribution(ga, nodeid)
@tst lo[1] == ((nodeid-1)*5)+1
@tst hi[1] == lo[1]+4

nputs(nodeid, string(lo, "-", hi))

# write into the local part
p = access(ga, lo, hi)
for i = 1:hi[1]-lo[1]+1
    p[i] = Aelem(nodeid, nodeid*nodeid)
end

# sync to let all nodes complete writing
sync(ga)

# get the whole array on node 1 and verify it
if nodeid == 1
    fa = get(ga, [1], [nelems])
    nputs(nodeid, fa)
end

#finalize(ga)

#=
# uneven distribution
# ---
nelems = nelems + (nnodes/2)
ga = Garray(Aelem, nelems)

# get the local part, write into it, and sync
lo, hi = distribution(ga, nodeid)
nputs(nodeid, string(lo, "-", hi))
p = access(ga, lo, hi)
for i = 1:hi[1]-lo[1]+1
    p[i] = Aelem(nodeid, nodeid*nodeid)
end
sync(ga)

# get the whole array on node 1 and verify it
if nodeid == 1
    fa = get(ga, [1], [nelems])
    nputs(nodeid, fa)
end
finalize(ga)
=#

