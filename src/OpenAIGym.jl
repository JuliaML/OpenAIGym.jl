
 __precompile__()

module OpenAIGym

using PyCall
using Reexport
@reexport using Reinforce
import Reinforce:
    MouseAction, MouseActionSet,
    KeyboardAction, KeyboardActionSet

export
    gym,
    GymEnv,
    test_env

const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

abstract type AbstractGymEnv <: AbstractEnvironment end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct GymEnv <: AbstractGymEnv
    name::String
    pyenv  # the python "env" object
    state
    reward::Float64
    total_reward::Float64
    actions::AbstractSet
    done::Bool
    info::Dict
    GymEnv(name,pyenv) = new(name,pyenv)
end
GymEnv(name) = gym(name)

function Reinforce.reset!(env::GymEnv)
    env.state = env.pyenv[:reset]()
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
end

function gym(name::AbstractString)
    env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        Base.copy!(gym_soccer, pyimport("gym_soccer"))
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name))
        end
    else
        GymEnv(name, pygym[:make](name))
    end
    reset!(env)
    env
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

function Reinforce.step!(env::GymEnv, a)
    # info("Going to take action: $a")
    pyact = pyaction(a)
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward = r
    env.total_reward += r
    env.state = s′
    r, s′
end

Reinforce.finished(env::GymEnv) = env.done
Reinforce.finished(env::GymEnv, s′) = env.done

# --------------------------------------------------------------


function test_env(name::String = "CartPole-v0")
    env = gym(name)
    for sars′ in Episode(env, RandomPolicy())
        render(env)
    end
end



global const pygym = PyNULL()
global const pysoccer = PyNULL()

function __init__()
    # the copy! puts the gym module into `pygym`, handling python ref-counting
    Base.copy!(pygym, pyimport("gym"))
end

end # module
