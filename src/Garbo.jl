module Garbo

using Base.Threads
enter_gc_safepoint() = ccall(:jl_gc_safe_enter, Int8, ())
leave_gc_safepoint(gs) = ccall(:jl_gc_safe_leave, Void, (Int8,), gs)

import Base.ndims, Base.length, Base.size, Base.get, Base.put!, Base.flush
export Garray, GarrayMemoryHandle, Dtree, nnodes, nodeid,
       sync, distribution, access,
       initwork, getwork, runtree

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
        global exiting
        exiting = true
    end
end

function __shutdown__()
    ccall((:garbo_shutdown, libgarbo), Void, (Ptr{Void},), ghandle[1])
end

@inline sync() = ccall((:garbo_sync, libgarbo), Void, ())
@inline cpu_pause() = ccall((:cpu_pause, libgarbo), Void, ())
@inline rdtsc() = ccall((:rdtsc, libgarbo), Culonglong, ())

include("Garray.jl")
include("Dtree.jl")

end # module

