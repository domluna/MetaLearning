
"""
Reptile: A Scalable Meta-Learning Algorithm [1]

[1] https://blog.openai.com/reptile/
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

        # Train on task for k steps on the dataset
        y = task(x)
        for idx in partition(randperm(length(x)), train_batch_size)
            l = Flux.mse(model(x[idx]'), y[idx]')
            Flux.back!(l)
            Flux.Optimise._update_params!(inner_opt, weights)
        end

        # Reptile update
        for (w1, w2) in zip(weights, prev_weights)
            gw = Flux.Optimise.apply!(meta_opt, w2, w1.data - w2)
            @. w1.data = w2 + gw
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end

    end
end

