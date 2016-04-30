# OpenAIGym

[![Build Status](https://travis-ci.org/tbreloff/OpenAIGym.jl.svg?branch=master)](https://travis-ci.org/tbreloff/OpenAIGym.jl)

#### Author: Thomas Breloff (@tbreloff)

This wraps the open source python library `gym`, released by OpenAI.  See [their website](https://gym.openai.com/) for more information.  Collaboration welcome!

### Setup

First install `gym`. Follow the instructions [here](https://gym.openai.com/docs).

Then add this julia package:

```julia
Pkg.clone("https://github.com/tbreloff/OpenAIGym.jl.git")
using OpenAIGym
```

### Hello world!

```julia
env = Env("CartPole-v0")
for i=1:20
    state = reset(env)
    for t in 1:100
        display(env)
        @show state
        action = rand(env)  # choose wisely!
        state = step(env, action)
        if state.done
            info("Episode finished after $t timesteps")
            break
        end
    end
end
```

Note: this is equivalent to the python code:
```python
import gym
env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
```


If everything works, a gui window will pop up with a tipping cart pole, and you should see output like:

```
state = OpenAIGym.State([0.009976859606990074,0.03900085260016814,-0.002189964775710318,0.036814067120848074],0.0,false,nothing)
state = OpenAIGym.State([0.010756876658993436,-0.1560896258540665,-0.0014536834332933564,0.32880523394055744],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([0.0076350841419121065,-0.3511908523164852,0.005122421245517792,0.6210293816400967],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([0.0006112670955824018,-0.5463839631030311,0.01754300887831973,0.9153211918052833],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.010316412166478219,-0.35150360217061205,0.035849432714425394,0.6282030111868846],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.01734648420989046,-0.5471070605423407,0.048413492938163084,0.9319571552127409],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.028288625420737276,-0.3526706578666229,0.0670526360424179,0.6548722914543239],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.03534203857806974,-0.15854319447979903,0.08015008187150438,0.38403416327769446],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.03851290246766572,0.035354776822726364,0.08783076513705827,0.11765981463376579],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.037805806931211196,0.2291156703886682,0.09018396142973359,-0.14607214928009798],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.033223493523437835,0.03282572278516144,0.08726251844413163,0.17364440096592187],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.032566979067734605,-0.1634297218094521,0.09073540646345007,0.49253008542318205],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.03583557350392365,-0.35970646164662734,0.10058600817191371,0.8123737000434847],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.0430297027368562,-0.16609554599411436,0.1168334821727834,0.5529476838018288],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.04635161365673848,0.027208627259268453,0.12789243584881999,0.29923908705344887],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.04580744111155311,-0.1694822893754317,0.13387721758988896,0.6293638298255058],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.04919708689906174,-0.366193356983638,0.1464644941863991,0.9610325548193519],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.0565209540387345,-0.5629477774976097,0.16568514528278613,1.2959110525878323],1.0,false,Dict{Any,Any}())
state = OpenAIGym.State([-0.0677799095886867,-0.7597407410478013,0.19160336633454278,1.635546922658331],1.0,false,Dict{Any,Any}())
INFO: Episode finished after 19 timesteps
```
