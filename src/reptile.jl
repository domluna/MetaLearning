
"""
"""
function reptile(model; inner_opt=Descent(0.02), epochs=30_000, 
                 train_batch_size=10, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = range(-5, stop=5, length=50)
    x = testx

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))
        task = SineWave()

        y = task(x)
        for idx in partition(randperm(length(x)), train_batch_size)
            l = Flux.mse(model(x[idx]'), y[idx]')
            Flux.back!(l)
            Flux.Optimise.update!(inner_opt, weights)
        end

        meta_stepsize = 0.1f0 * (1f0 - i / epochs)
        for (w1, w2) in zip(weights, prev_weights)
            @. w1.data = w2 + (w1.data - w2) * meta_stepsize
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_x = rand(dist, eval_batch_size)
            eval_model(model, eval_x, testx, SineWave())
        end

    end
end

