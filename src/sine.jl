
struct SineTask
    amplitude::Float64
    phase_shift::Float64
    interval::Uniform{Float64}
end

SineTask(amplitude, phase_shift) = SineTask(amplitude, phase_shift, Uniform(-5, 5))
SineTask() = SineTask(rand(Uniform(0.1, 5)), rand(Uniform(0, 2pi)))

function minibatch(m::SineTask, n::Int)
    x = rand(m.interval, n)
    y = m.amplitude .* sin.(x .+ m.phase_shift)
    return x, y
end

function test_set(m::SineTask, p=50, interval=m.interval)
    x = LinRange(interval.a, interval.b, p) |> Array
    y = m.amplitude .* sin.(x .+ m.phase_shift)
    return x, y
end
