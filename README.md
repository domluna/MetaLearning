# MetaLearning

> This has been moved to the [`model-zoo`](https://github.com/FluxML/model-zoo/blob/master/contrib/meta-learning/MetaLearning.jl).

Meta-learning sine waves. See [this post](https://www.domluna.com/meta-learning/) for details.

Example usage:

```julia
using MetaLearning, Flux

m = Chain(Linear(1, 64, tanh), Linear(64, 64, tanh), Linear(64, 1))

fomaml(m, epochs=50_000)
# OR
reptile(m, epochs=50_000)
```
