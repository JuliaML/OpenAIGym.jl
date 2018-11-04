module OpenAIGym

using PyCall
using Reexport
@reexport using Reinforce
import Reinforce:
    MouseAction, MouseActionSet,
    KeyboardAction, KeyboardActionSet

export
    pygym,
    GymEnv,
    test_env,
    PyAny

const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

abstract type AbstractGymEnv <: AbstractEnvironment end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct GymEnv{T} <: AbstractGymEnv
    name::String
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystate::PyObject # the state array object referenced by the PyArray state.o
    pystepres::PyObject # used to make stepping the env slightly more efficient
    pytplres::PyObject  # used to make stepping the env slightly more efficient
    info::PyObject    # store it as a PyObject for speed, since often unused
    state::T
    reward::Float64
    total_reward::Float64
    actions::AbstractSet
    done::Bool
    function GymEnv{T}(name, pyenv, pystate, state::T) where T
        env = new{T}(name, pyenv, pyenv["step"], pyenv["reset"],
                         pystate, PyNULL(), PyNULL(), PyNULL(), state)
        reset!(env)
        env
    end
end

function GymEnv(name; stateT=PyArray)
    env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        copy!(pysoccer, pyimport("gym_soccer"))
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name), stateT)
        end
    else
        GymEnv(name, pygym[:make](name), stateT)
    end
    env
end

function GymEnv(name, pyenv, stateT)
    pystate = pycall(pyenv["reset"], PyObject)
    state = convert(stateT, pystate)
    T = typeof(state)
    GymEnv{T}(name, pyenv, pystate, state)
end


# --------------------------------------------------------------

render(env::AbstractGymEnv, args...; kwargs...) =
    pycall(env.pyenv[:render], PyAny; kwargs...)

# --------------------------------------------------------------


function actionset(A::PyObject)
    if haskey(A, :n)
        # choose from n actions
        DiscreteSet(0:A[:n]-1)
    elseif haskey(A, :spaces)
        # a tuple of action sets
        sets = [actionset(a) for a in A[:spaces]]
        TupleSet(sets...)
    elseif haskey(A, :high)
        # continuous interval
        IntervalSet{Vector{Float64}}(A[:low], A[:high])
        # if A[:shape] == (1,)  # for now we only support 1-length vectors
        #     IntervalSet{Float64}(A[:low][1], A[:high][1])
        # else
        #     # @show A[:shape]
        #     lo,hi = A[:low], A[:high]
        #     # error("Unsupported shape for IntervalSet: $(A[:shape])")
        #     [IntervalSet{Float64}(lo[i], hi[i]) for i=1:length(lo)]
        # end
    elseif haskey(A, :actions)
        # Hardcoded
        TupleSet(DiscreteSet(A[:actions]))
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end

function Reinforce.actions(env::AbstractGymEnv, s′)
    actionset(env.pyenv[:action_space])
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a) = a

"""
`reset!` for PyArray state types
"""
function Reinforce.reset!(env::GymEnv{T}) where T <: PyArray
    setdata!(env.state, pycall!(env.pystate, env.pyreset, PyObject))
    return gymreset!(env)
end

"""
`reset!` for non PyArray state types
"""
function Reinforce.reset!(env::GymEnv{T}) where T
    pycall!(env.pystate, env.pyreset, PyObject)
    env.state = convert(T, env.pystate)
    return gymreset!(env)
end

function gymreset!(env::GymEnv{T}) where T
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
    return env.state
end

"""
`step!` for PyArray state
"""
function Reinforce.step!(env::GymEnv{T}, a) where T <: PyArray
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    unsafe_gettpl!(env.pystate, env.pystepres, PyObject, 0)
    setdata!(env.state, env.pystate)

    return gymstep!(env)
end

"""
step! for non-PyArray state
"""
function Reinforce.step!(env::GymEnv{T}, a) where T
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    unsafe_gettpl!(env.pystate, env.pystepres, PyObject, 0)
    env.state = convert(T, env.pystate)

    return gymstep!(env)
end

@inline function gymstep!(env)
    r = unsafe_gettpl!(env.pytplres, env.pystepres, Float64, 1)
    env.done = unsafe_gettpl!(env.pytplres, env.pystepres, Bool, 2)
    unsafe_gettpl!(env.info, env.pystepres, PyObject, 3)
    env.total_reward += r
    return (r, env.state)
end

Reinforce.finished(env::GymEnv)     = env.done
Reinforce.finished(env::GymEnv, s′) = env.done

# --------------------------------------------------------------

const pygym    = PyNULL()
const pysoccer = PyNULL()

function __init__()
    # the copy! puts the gym module into `pygym`, handling python ref-counting
    copy!(pygym, pyimport("gym"))
end

end # module
