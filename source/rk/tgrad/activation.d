module rk.tgrad.activation;

import rk.tgrad.value;

import std.math : exp, tanh;
import std.algorithm : min, max;

auto activateSigmoid(T)(Value!T x) 
{
    auto result = value(1 / (1 + exp(-x)), [x]);
    result._backward = () 
    {
        result.grad += (result.data * (1 - result.data)) * result.grad;
    };

    return result;
}

auto activateTanh(T)(Value!T x) 
{
    auto result = value(tanh(x), [x]);
    result._backward = () 
    {
        result.grad += (1 - (result.data * result.data)) * result.grad;
    };

    return result;
}

auto activateRelu(T)(Value!T x) 
{
    auto result = value(max(0, x), [x]);
    result._backward = () 
    {
        result.grad += (result.data > 0) * result.grad;
    };

    return result;
}

auto activateLeakyRelu(T)(Value!T x) 
{
    auto result = value(max(0, x) + 0.01 * min(0, x), [x]);
    result._backward = () 
    {
        result.grad += (result.data >= 0 ? 1 : 0.01) * result.grad;
    };

    return result;
}

