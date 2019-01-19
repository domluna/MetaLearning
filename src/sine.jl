
struct SineWave
    amplitude::Float64
    phase_shift::Float64
end
SineWave() = SineWave(rand(Uniform(0.1, 5)), rand(Uniform(0, 2pi)))

(s::SineWave)(x::AbstractArray) = s.amplitude .* sin.(x .+ s.phase_shift)

function Base.show(io::IO, s::SineWave)
    print(io, "SineWave(amplitude = ", s.amplitude, ", phase shift = ", s.phase_shift, ")")
end
