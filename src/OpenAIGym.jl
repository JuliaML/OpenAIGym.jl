
 __precompile__()

module OpenAIGym

using PyCall

export
    gym,
    Env,
    State,
    action_space

# --------------------------------------------------------------

type Env
    name
    env
end
Env(name::AbstractString) = Env(name, gym.make(name))

# --------------------------------------------------------------

"show the state (gui or whatever is supported)"
Base.display(env::Env) = env.env[:render]()

action_space(env::Env) = env.env[:action_space]

# --------------------------------------------------------------

immutable State
    observation
    reward
    done
    info
end

# --------------------------------------------------------------

"initializes a new episode"
Base.reset(env::Env) = State(env.env[:reset](), 0.0, false, nothing)


"choose a random action"
Base.rand(env::Env) = action_space(env)[:sample]()


"returns a State object"
Base.step(env::Env, action) = State(env.env[:step](action)...)

# --------------------------------------------------------------

# todo: iterate through an episode with something like:
#   for (observation, reward, info) in env
#       # choose an action
#       act(env, action)
#   end

# todo: standardize the loop above with a macro `@episode`



function __init__()
    global const gym = PyCall.pywrap(PyCall.pyimport("gym"))
end

end # module
