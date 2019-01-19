
function transfer_learn(model; opt=Descent(1e-2), epochs=30_000, 
                        train_batch_size=50, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = range(-5, stop=5, length=50)

    for i in 1:epochs
        task = SineWave()
        x = rand(dist, train_batch_size)
        y = task(x)
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise.update!(opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_x = rand(dist, eval_batch_size)
            eval_model(model, eval_x, testx, SineWave())
        end
    end
end
