# OpenAIGym

[![Build Status](https://travis-ci.org/tbreloff/OpenAIGym.jl.svg?branch=master)](https://travis-ci.org/tbreloff/OpenAIGym.jl)

#### Author: Thomas Breloff (@tbreloff)

This wraps the open source python library `gym`, released by OpenAI.  See [their website](https://gym.openai.com/) for more information.  Collaboration welcome!

### Setup

First install `gym`. Follow the instructions [here](https://gym.openai.com/docs).

Then add this julia package:

```julia
Pkg.clone("https://github.com/tbreloff/OpenAIGym.jl.git")
```

until it's registered, you'll also need to manually install [Reinforce.jl](https://github.com/tbreloff/Reinforce.jl):

```julia
Pkg.clone("https://github.com/tbreloff/Reinforce.jl.git")
```

---

### Hello world!

```julia
using OpenAIGym
env = GymEnv("CartPole-v0")
for i=1:20
    R, T = episode!(env, RandomPolicy())
    info("Episode $i finished after $T steps. Total reward: $R")
end
```

If everything works you should see output like this:

```
INFO: Episode 1 finished after 10 steps. Total reward: 10.0
INFO: Episode 2 finished after 46 steps. Total reward: 46.0
INFO: Episode 3 finished after 14 steps. Total reward: 14.0
INFO: Episode 4 finished after 19 steps. Total reward: 19.0
INFO: Episode 5 finished after 15 steps. Total reward: 15.0
INFO: Episode 6 finished after 32 steps. Total reward: 32.0
INFO: Episode 7 finished after 36 steps. Total reward: 36.0
INFO: Episode 8 finished after 13 steps. Total reward: 13.0
INFO: Episode 9 finished after 62 steps. Total reward: 62.0
INFO: Episode 10 finished after 14 steps. Total reward: 14.0
INFO: Episode 11 finished after 14 steps. Total reward: 14.0
INFO: Episode 12 finished after 28 steps. Total reward: 28.0
INFO: Episode 13 finished after 21 steps. Total reward: 21.0
INFO: Episode 14 finished after 15 steps. Total reward: 15.0
INFO: Episode 15 finished after 12 steps. Total reward: 12.0
INFO: Episode 16 finished after 20 steps. Total reward: 20.0
INFO: Episode 17 finished after 19 steps. Total reward: 19.0
INFO: Episode 18 finished after 17 steps. Total reward: 17.0
INFO: Episode 19 finished after 35 steps. Total reward: 35.0
INFO: Episode 20 finished after 23 steps. Total reward: 23.0
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

We're using the `RandomPolicy` from Reinforce.jl.  To do something better, you can create your own policy simply by implementing the `action` method, which takes a reward, a state, and an action set:

```julia
type RandomPolicy <: AbstractPolicy end
Reinforce.action(policy::AbstractPolicy, r, s, A) = rand(A)
```

Note: You can override default behavior of in the `episode!` method by overriding `Reinforce.on_step(env, i)`.
