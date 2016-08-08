module Garbo

#if VERSION > v"0.5.0-dev"
if isdefined(Base, :Threads)
    using Base.Threads
    enter_gc_safepoint() = ccall(:jl_gc_safe_enter, Int8, ())
    leave_gc_safepoint(gs) = ccall(:jl_gc_safe_leave, Void, (Int8,), gs)
else
    # Pre-Julia 0.5 there are no threads
    nthreads() = 1
    threadid() = 1
    enter_gc_safepoint() = 1
    leave_gc_safepoint(gs) = 1
end

export Garray, Dtree, nnodes, nodeid,
       get, put, sync, distribution, access,
       initwork, getwork, runtree

const fan_out = 2048
const drain_rate = 0.4
const libgarbo = joinpath(dirname(@__FILE__), "..", "deps", "Garbo",
        "libgarbo.$(Libdl.dlext)")

function __init__()
    global const ghandle = [ 0 ]
    ccall((:garbo_init, libgarbo), Cint, (Cint, Ptr{Ptr{UInt8}}, Ptr{Void}),
          length(ARGS), ARGS, pointer(ghandle, 1))
    global const nnodes = Int(ccall((:garbo_nnodes, libgarbo), Cint, ()))
    global const nodeid = Int(ccall((:garbo_nodeid, libgarbo), Cint, ())+1)
    atexit() do
        ccall((:garbo_shutdown, libgarbo), Cint, (Ptr{Void},), ghandle[1])
    end
end

type Garray
    ahandle::Array{Ptr{Void}}

    function Garray(T::DataType, dims::Array{Int}; chunks::Array{Int}=[])
        a = new([0])
        r = ccall((:garbo_create, libgarbo), Cint, (Ptr{Void}, Cint, Ptr{Cint},
                Cint, Ptr{Cint}, Ptr{Void}), ghandle[1], length(dims),
                dims, sizeof(T), chunks, pointer(a.ahandle, 1))
        if r != 0
            error("construction failure")
        end
        finalizer(a, (x -> ccall((:garbo_destroy, libgarbo),
                                 Void, (Ptr{Void},), a.ahandle[1])))
        return a
    end
end

function get()
end

sync() = ccall((:garbo_sync, libgarbo), Void, (Ptr{Void},), ghandle[1])

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
    r = ccall((:garbo_initwork, libgarbo), Cint,
            (Ptr{Void}, Ptr{Int64}, Ptr{Int64}), dt.handle[1], wp1, wp2)
    leave_gc_safepoint(gs)
    return r, (w[1]+1, w[2])
end

function getwork(dt::Dtree)
    w = [ 1, 1 ]::Array{Int64}
    wp1 = pointer(w, 1)
    wp2 = pointer(w, 2)
    gs = enter_gc_safepoint()
    r = ccall((:garbo_getwork, libgarbo), Cint,
            (Ptr{Void}, Ptr{Int64}, Ptr{Int64}), dt.handle[1], wp1, wp2)
    leave_gc_safepoint(gs)
    return r, (w[1]+1, w[2])
end

function runtree(dt::Dtree)
    r = 0
    gs = enter_gc_safepoint()
    r = ccall((:garbo_run, libgarbo), Cint, (Ptr{Void},), dt.handle[1])
    leave_gc_safepoint(gs)
    Bool(r > 0)
end

@inline cpu_pause() = ccall((:cpu_pause, libgarbo), Void, ())
@inline rdtsc() = ccall((:rdtsc, libgarbo), Culonglong, ())

end # module

