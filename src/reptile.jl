function reptile(model; opt=Descent(1e-2), inner_opt=Descent(0.02), epochs=10_000, grad_steps=5, batch_size=10, eval_interval=1000)
    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))
    eval_task = SineTask()

    for i in 1:epochs
        for (w1, w2) in zip(weights, prev_weights)
            w2 .= w1.data
        end

        task = SineTask()

        for _ in 1:grad_steps
            x, y = minibatch(task, batch_size)
            l = Flux.mse(model(x'), y')
            Flux.back!(l)
            Flux.Optimise.update!(inner_opt, weights)
        end

        #= x, y = test_set(task) =#
        #= for idx in partition(randperm(length(x)), batch_size) =#
        #=     l = Flux.mse(model(x[idx]'), y[idx]') =#
        #=     Flux.back!(l) =#
        #=     Flux.Optimise.update!(inner_opt, weights) =#
        #= end =#

        meta_stepsize = Float32(0.1) * (1 - i / epochs)
        for (w1, w2) in zip(weights, prev_weights)
            @. w1.data = w2 + (w1.data - w2) * meta_stepsize
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_model(model, eval_task)
        end

    end
end

