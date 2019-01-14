module MetaLearning

using Flux
using Distributions: Uniform
using Statistics: mean
using Printf
using Plots
using Base.Iterators: partition
using Random: randperm

export SineTask, maml, reptile, eval_model

"""
"""
function eval_model(model, task=SineTask(); batch_size=5, updates=10)
    opt = ADAM(1e-2)

    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))

    x, y = test_set(task, batch_size, Uniform(-5, 5))
    test_x, test_y = test_set(task)

    init_preds = model(test_x')
    test_loss = Flux.mse(init_preds, test_y')
    @printf("After 0 fits, Loss = %f\n", test_loss)

    for i in 1:updates
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise.update!(opt, weights)
        test_loss = Flux.mse(model(test_x'), test_y')
        @printf("After %d fits, Loss = %f\n", i, test_loss)
    end

    final_preds = model(test_x')

    for (w1, w2) in zip(weights, prev_weights)
        w1.data .= w2
        w1.grad .= 0
    end

    return x, y, test_x, test_y, Array(Flux.data(init_preds)'), Array(Flux.data(final_preds)')
end

include("sine.jl")
include("maml.jl")
include("reptile.jl")

end # module
