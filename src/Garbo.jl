module Garbo

using Base.Threads
enter_gc_safepoint() = ccall(:jl_gc_safe_enter, Int8, ())
leave_gc_safepoint(gs) = ccall(:jl_gc_safe_leave, Void, (Int8,), gs)

import Base.ndims, Base.length, Base.size, Base.get, Base.put!
export Garray, Dtree, nnodes, nodeid,
       flush, sync, distribution, access,
       initwork, getwork, runtree

const fan_out = 2048
const drain_rate = 0.4
const libgarbo = joinpath(dirname(@__FILE__), "..", "deps", "Garbo",
        "libgarbo.$(Libdl.dlext)")

function __init__()
    global const ghandle = [C_NULL]
    ccall((:garbo_init, libgarbo), Int64, (Cint, Ptr{Ptr{UInt8}}, Ptr{Void}),
          length(ARGS), ARGS, pointer(ghandle, 1))
    global const nnodes = ccall((:garbo_nnodes, libgarbo), Int64, ())
    global const nodeid = ccall((:garbo_nodeid, libgarbo), Int64, ())+1
    global num_garrays = 0
    global exiting = false
    atexit() do
        exiting = true
    end
end

function __shutdown__()
    ccall((:garbo_shutdown, libgarbo), Void, (Ptr{Void},), ghandle[1])
end

function sync()
    ccall((:garbo_sync, libgarbo), Void, ())
end

type Garray
    ahandle::Array{Ptr{Void}}
    atyp::DataType
end

function Garray(T::DataType, dims...)
    nd = length(dims)
    if nd < 1
        error("Garray must have at least one dimension")
    end
    adims = collect(dims)::Vector{Int64}
    a = Garray([C_NULL], T)
    r = ccall((:garray_create, libgarbo), Int64, (Ptr{Void}, Int64, Ptr{Int64},
            Int64, Ptr{Int64}, Ptr{Void}), ghandle[1], nd, adims, sizeof(T),
            C_NULL, pointer(a.ahandle, 1))
    if r != 0
        error("construction failure")
    end
    global num_garrays
    num_garrays = num_garrays+1
    finalizer(a, (function(a)
                    ccall((:garray_destroy, libgarbo),
                            Void, (Ptr{Void},), a.ahandle[1])
                    num_garrays = num_garrays-1
                    exiting && num_garrays == 0 && __shutdown__()
                  end))
    return a
end

function ndims(ga::Garray)
    ccall((:garray_ndims, libgarbo), Int64, (Ptr{Void},), ga.ahandle[1])
end

function length(ga::Garray)
    ccall((:garray_length, libgarbo), Int64, (Ptr{Void},), ga.ahandle[1])
end

function size(ga::Garray)
    dims = Array(Int64, ndims(ga))
    r = ccall((:garray_size, libgarbo), Int64, (Ptr{Void}, Ptr{Int64}),
            ga.ahandle[1], dims)
    if r != 0
        error("could not get size")
    end
    return tuple(dims...)
end

function flush(ga::Garray)
    ccall((:garray_sync, libgarbo), Void, (Ptr{Void},), ga.ahandle[1])
end

function get(ga::Garray, lo::Vector{Int64}, hi::Vector{Int64})
    adjlo = lo - 1
    adjhi = hi - 1
    dims = hi - lo + 1
    buf = Array(ga.atyp, dims...)
    r = ccall((:garray_get, libgarbo), Int64, (Ptr{Void}, Ptr{Int64}, Ptr{Int64},
            Ptr{Void}), ga.ahandle[1], adjlo, adjhi, buf)
    if r != 0
        error("Garray get failed")
    end
    return buf
end

function put!(ga::Garray, lo::Vector{Int64}, hi::Vector{Int64}, buf::Array)
end

function distribution(ga::Garray, nid::Int64)
    nd = ndims(ga)
    lo = Array(Int64, nd)
    hi = Array(Int64, nd)
    r = ccall((:garray_distribution, libgarbo), Int64, (Ptr{Void}, Int64,
            Ptr{Int64}, Ptr{Int64}), ga.ahandle[1], nid-1, lo, hi)
    if r != 0
        error("could not get distribution")
    end
    lo = lo+1
    hi = hi+1
    return lo, hi
end

function access(ga::Garray, lo::Vector{Int64}, hi::Vector{Int64})
    p = [C_NULL]
    r = ccall((:garray_access, libgarbo), Int64, (Ptr{Void}, Ptr{Int64},
            Ptr{Int64}, Ptr{Ptr{Void}}), ga.ahandle[1], lo-1, hi-1,
            pointer(p, 1))
    if r != 0
        error("could not get access")
    end
    return unsafe_wrap(Array, convert(Ptr{ga.atyp}, p[1]), hi[1]-lo[1]+1)
end


type Dtree
    handle::Array{Ptr{Void}}

    function Dtree(fan_out::Int, num_work_items::Int64,
            can_parent::Bool, node_mul::Float64,
            first::Float64, rest::Float64, min_dist::Int)
        parents_work = nthreads()>1 ? 1 : 0
        cthrid = cfunction(threadid, Int64, ())
        d = new([0])
        p = [ 0 ]
        r = ccall((:dtree_create, libgarbo), Cint,
                (Ptr{Void}, Cint, Cint, Cint, Cint, Cdouble, Cint, Ptr{Void},
                Cdouble, Cdouble, Cshort, Ptr{Void}, Ptr{Int64}),
                ghandle[1], fan_out, num_work_items, can_parent, parents_work,
                node_mul, nthreads(), cthrid, first, rest, min_dist,
                pointer(d.handle), pointer(p, 1))
        if r != 0
            error("construction failure")
        end
        finalizer(d, (x -> ccall((:dtree_destroy, libgarbo),
                                 Void, (Ptr{Void},), d.handle[1])))
        d, Bool(p[1])
    end
end

Dtree(num_work_items::Int64, first::Float64) =
    Dtree(fan_out, num_work_items, true, 1.0, first, drain_rate, 1)
Dtree(num_work_items::Int64, first::Float64, min_distrib::Int) =
    Dtree(fan_out, num_work_items, true, 1.0, first, drain_rate, min_distrib)

function initwork(dt::Dtree)
    w = [ 1, 1 ]::Array{Int64}
    wp1 = pointer(w, 1)
    wp2 = pointer(w, 2)
    gs = enter_gc_safepoint()
    r = ccall((:dtree_initwork, libgarbo), Cint,
            (Ptr{Void}, Ptr{Int64}, Ptr{Int64}), dt.handle[1], wp1, wp2)
    leave_gc_safepoint(gs)
    return r, (w[1]+1, w[2])
end

function getwork(dt::Dtree)
    w = [ 1, 1 ]::Array{Int64}
    wp1 = pointer(w, 1)
    wp2 = pointer(w, 2)
    gs = enter_gc_safepoint()
    r = ccall((:dtree_getwork, libgarbo), Cint,
            (Ptr{Void}, Ptr{Int64}, Ptr{Int64}), dt.handle[1], wp1, wp2)
    leave_gc_safepoint(gs)
    return r, (w[1]+1, w[2])
end

function runtree(dt::Dtree)
    r = 0
    gs = enter_gc_safepoint()
    r = ccall((:dtree_run, libgarbo), Cint, (Ptr{Void},), dt.handle[1])
    leave_gc_safepoint(gs)
    Bool(r > 0)
end

@inline cpu_pause() = ccall((:cpu_pause, libgarbo), Void, ())
@inline rdtsc() = ccall((:rdtsc, libgarbo), Culonglong, ())

end # module

