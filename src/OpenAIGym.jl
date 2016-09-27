
 __precompile__()

module OpenAIGym

using Reexport
using PyCall
@reexport using Reinforce

export
    gym,
    GymEnv

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
        env = new(name, gym[:make](name))
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


# returns a
function Reinforce.actions(env::GymEnv, s′)
    A = env.env[:action_space]
    if haskey(A, :n)
        DiscreteSet(0:A[:n]-1)
    else
        error()
    end
end

function Reinforce.step!(env::GymEnv, s, a)
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
