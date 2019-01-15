module MetaLearning

using Flux
using Distributions: Uniform
using Statistics: mean
using Printf
#= using Plots =#
using Base.Iterators: partition
using Random: randperm

export SineTask, transfer_learn, maml, reptile, eval_model

#= plot([x, test_x, test_x, test_x], [y, test_y, init_preds, final_preds], =#
#=              line=[:scatter :path :path :path], =#
#=              labels=["Sampled points", "Ground truth", "Before fitting", "After fitting"]) =#

"""
"""
function eval_model(model, task=SineTask(); opt=Descent(1e-2), batch_size=10, updates=32)
    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))

    #= x, y = test_set(task, batch_size, Uniform(-5, 0)) =#
    x, y = minibatch(task, batch_size)
    test_x, test_y = test_set(task)

    init_preds = model(test_x')
    test_loss = Flux.mse(init_preds, test_y')
    @printf("Task amplitude = %f, phase shift = %f\n", task.amplitude, task.phase_shift)
    @printf("Before finetuning, Loss = %f\n", test_loss)

    for i in 1:updates
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise.update!(opt, weights)
        test_loss = Flux.mse(model(test_x'), test_y')
        @printf("After %d fits, Loss = %f\n", i, test_loss)
    end
    final_preds = model(test_x')

    # reset weights
    for (w1, w2) in zip(weights, prev_weights)
        w1.data .= w2
        w1.grad .= 0
    end

    return x, test_x, y, test_y, Array(Flux.data(init_preds)'), Array(Flux.data(final_preds)')
end

include("sine.jl")
include("transfer.jl")
include("maml.jl")
include("reptile.jl")

end # module
