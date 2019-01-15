
function transfer_learn(model; opt=Descent(1e-2), epochs=30_000, batch_size=50, eval_interval=1000)
    weights = params(model)

    for i in 1:epochs
        task = SineTask()

        x, y = minibatch(task, batch_size)
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise.update!(opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            eval_model(model, SineTask())
        end
    end
end
