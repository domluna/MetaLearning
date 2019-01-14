function maml(model; opt=ADAM(1e-2), epochs=1000, n_tasks=5, batch_size=10, eval_interval=100)
    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))
    eval_task = SineTask()

    for i in 1:epochs
        # same as above deepcopy but inplace
        for (w1, w2) in zip(weights, prev_weights)
            w2 .= w1.data
        end

        loss = Float32(0)

        for _ in 1:n_tasks
            task = SineTask()
            x, y = minibatch(task, batch_size)

            grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights; nest=true)
            for w in weights
                w.data .-= Float32(0.02) * grad[w].data
            end

            x, y = test_set(task)
            loss += Flux.mse(model(x'), y')
            
            # reset weights
            for (w1, w2) in zip(weights, prev_weights)
                w1.data .= w2
                w1.grad .= 0
            end
        end

        Flux.back!(loss)
        Flux.Optimise.update!(opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on task...\n", i)
            eval_model(model, eval_task)
        end

    end
end

