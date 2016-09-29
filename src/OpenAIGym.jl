
 __precompile__()

module OpenAIGym


using Reexport
using PyCall
@reexport using Reinforce

export
    gym,
    GymEnv,
    test_env

const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type GymEnv <: AbstractEnvironment
    name::String
    pyenv  # the python "env" object
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

render(env::GymEnv, args...) = env.pyenv[:render]()

# --------------------------------------------------------------

function Reinforce.reset!(env::GymEnv)
    env.state = env.pyenv[:reset]()
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
        IntervalSet{Vector{Float64}}(A[:low], A[:high])
        # if A[:shape] == (1,)  # for now we only support 1-length vectors
        #     IntervalSet{Float64}(A[:low][1], A[:high][1])
        # else
        #     # @show A[:shape]
        #     lo,hi = A[:low], A[:high]
        #     # error("Unsupported shape for IntervalSet: $(A[:shape])")
        #     [IntervalSet{Float64}(lo[i], hi[i]) for i=1:length(lo)]
        # end
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end


function Reinforce.actions(env::GymEnv, s′)
    actionset(env.pyenv[:action_space])
end

function Reinforce.step!(env::GymEnv, s, a)
    # info("Going to take action: $a")
    s′, r, env.done, env.info = env.pyenv[:step](a)
    env.reward, env.state = r, s′
end

Reinforce.finished(env::GymEnv, s′) = env.done

# function Reinforce.on_step(env::GymEnv, i::Int)
#     # render(env)
# end

function test_env(name::String = "CartPole-v0")
    env = GymEnv(name)
    episode!(env, RandomPolicy(), stepfunc = render)
end


# --------------------------------------------------------------


function __init__()
    global const gym = pyimport("gym")
end

end # module
