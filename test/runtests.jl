using OpenAIGym
using Base.Test

# write your own tests here
@test 1 == 1


env = Env("CartPole-v0")
for i=1:1

    # initialize the episode
    state = reset(env)
    @show state

    # loop through timesteps
    for t in 1:100

        # update the view
        display(env)

        # random aciton
        action = rand(env)
        state = step(env, action)
        @show action, state

        if state.done
            info("Episode finished after $t timesteps")
            break
        end
    end
end


# similar to:
# -------------------------
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
