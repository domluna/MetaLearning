
"""
"""
function reptile(model; meta_opt=Descent(0.1), inner_opt=Descent(0.02), epochs=30_000, 
                 train_batch_size=10, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))
    x = testx

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))
        task = SineWave()

        y = task(x)
        for idx in partition(randperm(length(x)), train_batch_size)
        #= for _ in 1:5 =#
            #= x = rand(dist, train_batch_size) =#
            #= y = task(x) =#
            l = Flux.mse(model(x[idx]'), y[idx]')
            #= l = Flux.mse(model(x'), y') =#
            Flux.back!(l)
            Flux.Optimise.update!(inner_opt, weights)
        end

        for (w1, w2) in zip(weights, prev_weights)
            g = Flux.Optimise.update!(meta_opt, w2, w1.data - w2)
            @. w1.data = w2 + g
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end

    end
end

