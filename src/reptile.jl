
function reptile(model; opt=Descent(0.01), inner_opt=Descent(0.02), epochs=30_000, batch_size=10, eval_interval=1000)
    weights = params(model)

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))
        task = SineTask()

        x, y = test_set(task)
        for idx in partition(randperm(length(x)), batch_size)
            l = Flux.mse(model(x[idx]'), y[idx]')
            Flux.back!(l)
            Flux.Optimise.update!(inner_opt, weights)
        end

        meta_stepsize = Float32(0.1) * (1 - i / epochs)
        for (w1, w2) in zip(weights, prev_weights)
            @. w1.data = w2 + (w1.data - w2) * meta_stepsize
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_model(model, SineTask())
        end

    end
end

