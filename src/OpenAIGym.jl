
 __precompile__()

module OpenAIGym


using Reexport
using PyCall
@reexport using Reinforce

export
    gym,
    GymEnv

const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type GymEnv <: AbstractEnvironment
    name::String
    env
    state
    reward::Float64
    actions::AbstractSet
    done::Bool
    info::Dict
    function GymEnv(name::AbstractString)
        env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
            @pyimport gym_soccer
            get!(_py_envs, name) do
                new(name, gym[:make](name))
            end
        else
            new(name, gym[:make](name))
        end
        reset!(env)
        env
    end
end

# --------------------------------------------------------------

render(env::GymEnv, args...) = env.env[:render]()

# --------------------------------------------------------------

function Reinforce.reset!(env::GymEnv)
    env.state = env.env[:reset]()
    env.reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
end

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
        if A[:shape] == (1,)  # for now we only support 1-length vectors
            IntervalSet(A[:low][1], A[:high][1])
        else
            @show A[:shape]
            error("Unsupported shape for IntervalSet: $(A[:shape])")
        end
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end


function Reinforce.actions(env::GymEnv, s′)
    actionset(env.env[:action_space])
end

function Reinforce.step!(env::GymEnv, s, a)
    # info("Going to take action: $a")
    s′, r, env.done, env.info = env.env[:step](a)
    env.reward, env.state = r, s′
end

Reinforce.finished(env::GymEnv, s′) = env.done

# function Reinforce.on_step(env::GymEnv, i::Int)
#     # render(env)
# end

# --------------------------------------------------------------


function __init__()
    global const gym = pyimport("gym")
end

end # module
