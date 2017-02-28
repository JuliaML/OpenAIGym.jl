using OpenAIGym
using Base.Test

# write your own tests here
@test 1 == 1

if isinteractive()
    env = GymEnv("CartPole-v0")
    for i=1:5
        R = run_episode(()->nothing, env, RandomPolicy())
        info("Episode $i finished. Total reward: $R")
    end
end
