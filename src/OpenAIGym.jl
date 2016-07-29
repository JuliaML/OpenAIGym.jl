
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
    # should_reset::Bool
    state
    reward::Float64
    actions::AbstractActionSet
    info::Dict
    function GymEnv(name::AbstractString)
        env = new(name, gym.make(name)) #, true)
        reset!(env)
        env
    end
end

# --------------------------------------------------------------

render(env::GymEnv) = env.env[:render]()

# --------------------------------------------------------------

function Reinforce.reset!(env::GymEnv)
    # env.should_reset = true
    env.state = env.env[:reset]()
    env.reward = 0.0
    # env.should_reset = false
    env.actions = actions(env)
end


# returns a
function Reinforce.actions(env::GymEnv)
    A = env.env[:action_space]
    if haskey(A, :n)
        DiscreteActionSet(0:A[:n]-1)
    else
        error()
    end
end

function Reinforce.step!(env::GymEnv, policy::AbstractPolicy = RandomPolicy())
    # if env.should_reset
    #     # reset the episode
    #     env.state = env.env[:reset]()
    #     env.reward = 0.0
    #     env.should_reset = false
    #     env.actions = actions(env)
    # end

    # get an action from the policy
    a = action(policy, env.reward, env.state, env.actions)

    # apply the action and get the updated state/reward
    env.state, env.reward, done, env.info = env.env[:step](a)
    done
end


function Reinforce.on_step(env::GymEnv, i::Int)
    # render(env)
end

Reinforce.reward(env::GymEnv) = env.reward
Reinforce.reward!(env::GymEnv) = env.reward
Reinforce.state(env::GymEnv) = env.state
Reinforce.state!(env::GymEnv) = env.state

# --------------------------------------------------------------


function __init__()
    global const gym = PyCall.pywrap(PyCall.pyimport("gym"))
end

end # module
