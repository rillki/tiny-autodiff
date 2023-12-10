module rk.tgrad.activation;

import rk.tgrad.value;

import std.math : exp, tanh;
import std.algorithm : min, max;

auto activateSigmoid(T)(Value!T x)
{
    auto result = value(1 / (1 + exp(-x)), [x]);
    result._backward = () 
    {
        x.grad += (result.data * (1 - result.data)) * result.grad;
    };

    return result;
}

auto activateTanh(T)(Value!T x) 
{
    auto result = value(tanh(x), [x]);
    result._backward = () 
    {
        x.grad += (1 - result.data * result.data) * result.grad;
    };

    return result;
}

auto activateRelu(T)(Value!T x) 
{
    auto result = value(max(0, x), [x]);
    result._backward = () 
    {
        x.grad += (result.data > 0) * result.grad;
    };

    return result;
}

auto activateLeakyRelu(T)(Value!T x) 
{
    auto result = value(max(0, x) + 0.01 * min(0, x), [x]);
    result._backward = () 
    {
        x.grad += (result.data >= 0 ? 1 : 0.01) * result.grad;
    };

    return result;
}

unittest
{
    import std.math : isClose;

    // inputs x1,x2
    auto x1 = value(2.0);
    auto x2 = value(0.0);

    // weights w1,w2 (synaptic strengths)
    auto w1 = value(-3.0);
    auto w2 = value(1.0);

    // bias
    auto b = value(7.0);
    
    // x1*w1 + x2*w2 + b
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    auto o = n.activateTanh();

    assert(o.data.isClose(0.76, 0.01));
    assert(o.grad == 0);

    o.backward();

    assert(o.grad == 1);
    assert(x1w1x2w2.grad.isClose(0.42, 0.01));
    assert(b.grad.isClose(0.42, 0.01));
    assert(w1.grad.isClose(0.84, 0.01));
    assert(w2.grad.isClose(0.00, 0.01));
    assert(x1.grad.isClose(-1.25, 0.01));
    assert(x2.grad.isClose(0.42, 0.01));
}

