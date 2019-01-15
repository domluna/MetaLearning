"""

Return the recommended gain value for the given nonlinearity function.
The values are as follows:

================= ====================================================
nonlinearity      gain
================= ====================================================
Tanh              5 / 3
ReLU              sqrt(2)
Leaky Relu        sqrt(2 / (1 + negative_slope}^2))
Linear / Identity 1
Conv{1,2,3}D      1
Sigmoid           1
================= ====================================================

Args:
    nonlinearity: the non-linear function (`nn.functional` name)
    param: optional parameter for the non-linear function
"""
function calculate_gain(f::String, negative_slope=0.01f0)
    if f == "tanh"
        return 5f0 / 3f0
    elseif f == "relu"
        return sqrt(2f0)
    elseif f == "leaky_relu"
        return sqrt(2f0 / (1f0 + negative_slope^2))
    end
    return 1f0
end

function calculate_fan_in_and_fan_out(dims...)
    len = length(dims)
    @assert len > 1

    if len == 2
        fan_in = dims[2]
        fan_out = dims[1]
    else
        num_input_fmaps = dims[2]
        num_output_fmaps = dims[1]
        receptive_field_size = len > 2 ? prod(dims[3:end]) : 1
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    end

    return fan_in, fan_out
end

function calculate_correct_fan(dims...; fan_in=true)
    d1, d2 = calculate_fan_in_and_fan_out(dims...)
    fan_in && (return d1)
    return d2
end




"""

Fills the input `Tensor` with values according to the method
described in "Delving deep into rectifiers: Surpassing human-level
performance on ImageNet classification" - He, K. et al. (2015), using a
uniform distribution [-bound, bound] where bound is ... 

Also known as He initialization.

Args:
    tensor: an n-dimensional `torch.Tensor`
    a: the negative slope of the rectifier used after this layer (0 for ReLU
        by default)
    mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
        preserves the magnitude of the variance of the weights in the
        forward pass. Choosing `fan_out` preserves the magnitudes in the
        backwards pass.
    nonlinearity: the non-linear function (`nn.functional` name),
        recommended to use only with 'relu' or 'leaky_relu' (default).
"""
function kaiming_uniform(dims...; a=sqrt(5), fan_in=true, f="leaky_relu")
    fan = calculate_correct_fan(dims...; fan_in=fan_in)
    gain = calculate_gain(f, a)
    std = gain / sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = sqrt(3) * std  
    @info bound
    return (rand(Float32, dims...) .* 2f0 .- 1f0) .* Float32(bound)
end



"""

Fills the input `Tensor` with values according to the method
described in "Delving deep into rectifiers: Surpassing human-level
performance on ImageNet classification" - He, K. et al. (2015), using a
normal distribution with mean 0 and a standard deviation of ...


Also known as He initialization.

Args:
    tensor: an n-dimensional `torch.Tensor`
    a: the negative slope of the rectifier used after this layer (0 for ReLU
        by default)
    mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
        preserves the magnitude of the variance of the weights in the
        forward pass. Choosing `fan_out` preserves the magnitudes in the
        backwards pass.
    nonlinearity: the non-linear function (`nn.functional` name),
        recommended to use only with 'relu' or 'leaky_relu' (default).
"""
function kaiming_normal(dims...; a=0f0, fan_in=true, f="leaky_relu")
    fan = calculate_correct_fan(dims...; fan_in=fan_in)
    gain = calculate_gain(f, a)
    std = gain / sqrt(fan)
    @info std
    return randn(Float32, dims...) .* Float32(std)
end
