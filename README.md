# OpenAIGym

[![Build Status](https://travis-ci.org/JuliaML/OpenAIGym.jl.svg?branch=master)](https://travis-ci.org/JuliaML/OpenAIGym.jl) [![Gitter](https://badges.gitter.im/reinforcejl/Lobby.svg)](https://gitter.im/reinforcejl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

#### Author: Thomas Breloff (@tbreloff)

This wraps the open source python library `gym`, released by OpenAI.
See [their website](https://gym.openai.com/) for more information.
Collaboration welcome!

---

### Installation

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/JuliaML/OpenAIGym.jl.git")
```

### Hello world!

```julia
using OpenAIGym
env = GymEnv(:CartPole, :v0)
for i ∈ 1:20
  T = 0
  R = run_episode(env, RandomPolicy()) do (s, a, r, s′)
    OpenAIGym.render(env)
    T += 1
  end
  @info("Episode $i finished after $T steps. Total reward: $R")
end
```

If everything works you should see output like this:

```
[ Info: Episode 1 finished after 10 steps. Total reward: 10.0
[ Info: Episode 2 finished after 46 steps. Total reward: 46.0
[ Info: Episode 3 finished after 14 steps. Total reward: 14.0
[ Info: Episode 4 finished after 19 steps. Total reward: 19.0
[ Info: Episode 5 finished after 15 steps. Total reward: 15.0
[ Info: Episode 6 finished after 32 steps. Total reward: 32.0
[ Info: Episode 7 finished after 36 steps. Total reward: 36.0
[ Info: Episode 8 finished after 13 steps. Total reward: 13.0
[ Info: Episode 9 finished after 62 steps. Total reward: 62.0
[ Info: Episode 10 finished after 14 steps. Total reward: 14.0
[ Info: Episode 11 finished after 14 steps. Total reward: 14.0
[ Info: Episode 12 finished after 28 steps. Total reward: 28.0
[ Info: Episode 13 finished after 21 steps. Total reward: 21.0
[ Info: Episode 14 finished after 15 steps. Total reward: 15.0
[ Info: Episode 15 finished after 12 steps. Total reward: 12.0
[ Info: Episode 16 finished after 20 steps. Total reward: 20.0
[ Info: Episode 17 finished after 19 steps. Total reward: 19.0
[ Info: Episode 18 finished after 17 steps. Total reward: 17.0
[ Info: Episode 19 finished after 35 steps. Total reward: 35.0
[ Info: Episode 20 finished after 23 steps. Total reward: 23.0
```


Note: this is equivalent to the python code:

```python
import gym
env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    total_reward = 0.0
    observation = env.reset()
    for t in xrange(100):
        # env.render()
        # print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print "Episode {} finished after {} timesteps. Total reward: {}".format(i_episode, t+1, total_reward)
            break
```


---

We're using the `RandomPolicy` from Reinforce.jl.  To do something better, you can create your own policy simply by implementing the `action` method, which takes a reward, a state, and an action set, then returns an action selection:

```julia
type RandomPolicy <: AbstractPolicy end
Reinforce.action(policy::AbstractPolicy, r, s, A) = rand(A)
```

Note: You can override default behavior of in the `run_episode` method.
Just iterate yourself:

```julia
ep = Episode(env, policy)
for (s, a, r, s′) in ep
    # do something special?
    OpenAIGym.render(env)
end
R = ep.total_reward
N = ep.niter
```

---

### Install gym

First install `gym`. Follow the instructions [here](https://gym.openai.com/docs) if you're using a system-wide python, or to use Conda.jl:

```julia
Pkg.add("PyCall")
withenv("PYTHON" => "") do
   Pkg.build("PyCall")
end
```

then install gym from the command line:

```
~/.julia/conda/3/bin/pip install 'gym[all]'
```

For additional environments, use a similar process.
For example, here's how I installed [Soccer](https://github.com/openai/gym-soccer) on my Ubuntu machine:

```
cd /opt
git clone https://github.com/LARG/HFO
cd HFO
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
make install

cd /opt
git clone https://github.com/openai/gym-soccer
cd gym-soccer
~/.julia/conda/3/bin/pip install -e .
```
