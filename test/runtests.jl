using OpenAIGym
using PyCall
using Test

"""
`function time_steps(env::GymEnv{T}, num_eps::Int) where T`

run through num_eps eps, recording the time taken for each step and
how many steps were made. Doesn't time the `reset!` or the first step of each
episode (since higher chance that it's slower/faster than the rest, and we want
to compare the average time taken for each step as fairly as possible)
"""
function time_steps(env::GymEnv, num_eps::Int)
    t = 0.0
    steps = 0
    for i in 1:num_eps
        reset!(env)
        # step!(env, rand(env.actions)) # ignore the first step - it might be slow?
        t += (@elapsed steps += epstep(env))
    end
    steps, t
end

"""
Steps through an episode until it's `done`
assumes env has been `reset!`
"""
function epstep(env::GymEnv)
    steps = 0
    while true
        steps += 1
        r, s′ = step!(env, rand(env.actions))
        finished(env, s′) && break
    end
    steps
end

@testset "Gym Basics" begin

    pong = GymEnv("Pong-v4")
    pongnf = GymEnv("PongNoFrameskip-v4")
    pacman = GymEnv("MsPacman-v4")
    pacmannf = GymEnv("MsPacmanNoFrameskip-v4")
    cartpole = GymEnv("CartPole-v0")
    bj = GymEnv("Blackjack-v0", stateT=PyAny)

    allenvs = [pong, pongnf, pacman, pacmannf, cartpole, bj]
    eps2trial = Dict(pong=>2, pongnf=>1, pacman=>2, pacmannf=>1, cartpole=>400, bj=>30000)
    atarienvs = [pong, pongnf, pacman, pacmannf]
    envs = allenvs

    @testset "envs load" begin
        # check they all work - no errors == no worries
        println("------------------------------ Check envs load ------------------------------")
        for (i, env) in enumerate(envs)
            a = rand(env.actions) |> OpenAIGym.pyaction
            action_type = a |> PyObject |> pytypeof
            println("env.pyenv: $(env.pyenv) action_type: $action_type (e.g. $a)")
            time_steps(env, 1)
            @test !ispynull(env.pyenv)
            println("------------------------------")
        end
    end

    @testset "julia speed test" begin
        println("------------------------------ Begin Julia Speed Check ------------------------------")
        for env in envs
            num_eps = eps2trial[env]
            steps, t = time_steps(env, num_eps)
            println("env.pyenv: $(env.pyenv) num_eps: $num_eps t: $t steps: $steps")
            println("microsecs/step (lower is better): ", t*1e6/steps)
            println("------------------------------")
        end
        println("------------------------------ End Julia Speed Check ------------------------------\n")
    end

    @testset "python speed test" begin
        println("------------------------------ Begin Python Speed Check ------------------------------")
        py"""
        import gym
        import numpy as np

        pong = gym.make("Pong-v4")
        pongnf = gym.make("PongNoFrameskip-v4")
        pacman = gym.make("MsPacman-v4");
        pacmannf = gym.make("MsPacmanNoFrameskip-v4");
        cartpole = gym.make("CartPole-v0")
        bj = gym.make("Blackjack-v0")

        allenvs = [pong, pongnf, pacman, pacmannf, cartpole, bj]
        eps2trial = {pong: 2, pongnf: 1, pacman: 2, pacmannf: 1, cartpole: 400, bj: 30000}
        atarienvs = [pong, pongnf, pacman, pacmannf];

        envs = allenvs

        import time
        class Timer(object):
            elapsed = 0.0
            def __init__(self, name=None):
                self.name = name

            def __enter__(self):
                self.tstart = time.time()

            def __exit__(self, type, value, traceback):
                Timer.elapsed = time.time() - self.tstart

        def time_steps(env, num_eps):
            t = 0.0
            steps = 0
            for i in range(num_eps):
                env.reset()
                with Timer():
                    steps += epstep(env)
                t += Timer.elapsed
            return steps, t

        def epstep(env):
            steps = 0
            while True:
                steps += 1
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if done == True:
                    break
            return steps

        for env in envs:
            num_eps = eps2trial[env]
            with Timer():
                steps, s = time_steps(env, num_eps)
            t = Timer.elapsed
            print("{env} num_eps: {num_eps} t: {t} steps: {steps} \n"
                  "microsecs/step (lower is better): {time}".format(
                    env=env, num_eps=num_eps, t=t, steps=steps,
                    time=t*1e6/steps))
            print("------------------------------")
        """
        println("------------------------------ End Python Speed Check ------------------------------")
    end
end
