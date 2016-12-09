
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

abstract AbstractGymEnv <: AbstractEnvironment

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type GymEnv <: AbstractGymEnv
    name::String
    pyenv  # the python "env" object
    state
    reward::Float64
    actions::AbstractSet
    done::Bool
    info::Dict
    GymEnv(name,pyenv) = new(name,pyenv)
end
GymEnv(name) = gym(name)

function Reinforce.reset!(env::GymEnv)
    env.state = env.pyenv[:reset]()
    env.reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type UniverseEnv <: AbstractGymEnv
    name::String
    pyenv  # the python "env" object
    state
    reward
    actions::AbstractSet
    done
    info::Dict
    UniverseEnv(name,pyenv) = new(name,pyenv)
end
UniverseEnv(name) = gym(name)

function Reinforce.reset!(env::UniverseEnv)
    env.state = env.pyenv[:reset]()
    env.reward = [0.0]
    env.actions = actions(env, nothing)
    env.done = [false]
end

function gym(name::AbstractString)
    env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        @pyimport gym_soccer
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name))
        end
    elseif split(name, ".")[1] in ("flashgames", "wob")
        @pyimport universe
        @pyimport universe.wrappers as wrappers
        if !isdefined(OpenAIGym, :vnc_event)
            global const vnc_event = PyCall.pywrap(PyCall.pyimport("universe.spaces.vnc_event"))
        end
        get!(_py_envs, name) do
            pyenv = wrappers.SafeActionSpace(pygym[:make](name))
            pyenv[:configure](remotes=1)  # automatically creates a local docker container
            o = UniverseEnv(name, pyenv)
            # finalizer(o,  o.pyenv[:close]())
            o
        end
    else
        GymEnv(name, pygym[:make](name))
    end
    reset!(env)
    env
end


# --------------------------------------------------------------

render(env::AbstractGymEnv, args...) = env.pyenv[:render]()

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
    elseif haskey(A, :buttonmasks)
        # assumed VNC actions... keys to press, buttons to mask, and screen position
        # keyboard = DiscreteSet(A[:keys])
        keyboard = KeyboardActionSet(A[:keys])
        buttons = DiscreteSet(Int[bm for bm in A[:buttonmasks]])
        width,height = A[:screen_shape]
        mouse = MouseActionSet(width, height, buttons)
        TupleSet(keyboard, mouse)
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
pyaction(a::KeyboardAction) = Any[a.key]
pyaction(a::MouseAction) = Any[vnc_event.PointerEvent(a.x, a.y, a.button)]
pyaction(a) = a

function Reinforce.step!(env::GymEnv, s, a)
    info("Going to take action: $a")
    pyact = pyaction(a)
    if env.isuniverse
        pyact = Any[pyact]
    end
    @show pyact
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward, env.state = r, s′
end

function Reinforce.step!(env::UniverseEnv, s, a)
    info("Going to take action: $a")
    pyact = Any[pyaction(a)]
    @show pyact
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward, env.state = r, s′
end

Reinforce.finished(env::GymEnv, s′) = env.done
Reinforce.finished(env::UniverseEnv, s′) = all(env.done)

# function Reinforce.on_step(env::GymEnv, i::Int)
#     # render(env)
# end

function test_env(name::String = "CartPole-v0")
    env = gym(name)
    for sars′ in Episode(env, RandomPolicy())
        render(env)
    end
    # episode!(env, RandomPolicy(), stepfunc = render)
end


# --------------------------------------------------------------


function __init__()
    @static if is_linux()
        # due to a ssl library bug, I have to first load the ssl lib here
        condadir = Pkg.dir("Conda","deps","usr","lib")
        Libdl.dlopen(joinpath(condadir, "libssl.so"))
        Libdl.dlopen(joinpath(condadir, "python2.7", "lib-dynload", "_ssl.so"))
    end

    global const pygym = pyimport("gym")
end

end # module
