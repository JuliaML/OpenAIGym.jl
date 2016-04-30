
 __precompile__()

module OpenAIGym

using PyCall

type Env
    name::String
    env
end
Env(name::String) = Env(name, gym.make(name))

"initializes a new episode"
Base.reset(env::Env) = env.env[:reset]()

"show the state (gui or whatever is supported)"
Base.display(env::Env) = env.env[:render]()

action_space(env::Env) = env.env[:action_space]

"choose a random action"
Base.random(env::Env) = action_space(env)[:sample]()

"returns a tuple: (observation, reward, done, info)"
Base.step(env::Env, action) = env.env[:step](action)

# todo: iterate through an episode with something like:
#   for (observation, reward, info) in env
#       # choose an action
#       act(env, action)
#   end

# todo: standardize the loop above with a macro `@episode`

# an example:
# ----------------------
# import gym
# env = gym.make('CartPole-v0')
# for i_episode in xrange(20):
#     observation = env.reset()
#     for t in xrange(100):
#         env.render()
#         print observation
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print "Episode finished after {} timesteps".format(t+1)
#             break
# ----------------------

# the equivalent julia:
# ----------------------
using OpenAIGym
env = Env("CartPole-v0")
# ----------------------

function __init__()
    global const gym = PyCall.pywrap(PyCall.pyimport("gym"))
end

end # module
