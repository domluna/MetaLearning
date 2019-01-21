
"""
First-order MAML
"""
function maml(model; meta_opt=Descent(0.01), inner_opt=Descent(0.02), epochs=30_000, 
              n_tasks=3, train_batch_size=10, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))

        for _ in 1:n_tasks
            task = SineWave()

            x = Float32.(rand(dist, train_batch_size))
            y = task(x)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights)

            for w in weights
                g = Flux.Optimise.update!(inner_opt, w.data, grad[w].data)
                w.data .-= g
            end
            
            testy = task(testx)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(testx'), testy'), weights)

            # reset weights and accumulate gradients
            for (w1, w2) in zip(weights, prev_weights)
                w1.data .= w2
                w1.grad .+= grad[w1].data
            end
        end

        Flux.Optimise.update!(meta_opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end

    end
end

