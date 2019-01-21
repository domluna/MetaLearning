
function transfer_learn(model; opt=Descent(0.01), epochs=30_000, 
                        train_batch_size=50, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))

    for i in 1:epochs
        task = SineWave()
        x = Float32.(rand(dist, train_batch_size))
        y = task(x)
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise.update!(opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end
    end
end
