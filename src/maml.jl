"""
"""
function maml(model; opt=Descent(1e-2), epochs=30_000, 
              n_tasks=5, train_batch_size=10, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = range(-5, stop=5, length=50)

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))

        loss = 0f0
        for _ in 1:n_tasks
            task = SineWave()
            x = rand(dist, train_batch_size)
            y = task(x)

            #= grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights) =#
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights; nest=true)
            for w in weights
                w.data .-= 0.02f0 * grad[w].data
            end

            testy = task(testx)
            loss += Flux.mse(model(testx'), testy')
            
            # reset weights
            for (w1, w2) in zip(weights, prev_weights)
                w1.data .= w2
                w1.grad .= 0
            end
        end

        Flux.back!(loss)
        Flux.Optimise.update!(opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_x = rand(dist, eval_batch_size)
            eval_model(model, eval_x, testx, SineWave())
        end

    end
end

