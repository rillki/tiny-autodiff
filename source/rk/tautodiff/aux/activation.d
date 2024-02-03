module rk.tautodiff.aux.activation;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;

import std.math : exp, tanh;
import std.algorithm : min, max;

Value activateLinear(Value x) 
{
    return value(x.data, [x], (result) {
        auto _x = result.parents[0];
        _x.grad += 1 * result.grad;
    });
}

Value activateSigmoid(Value x)
{
    return value(1 / (1 + exp(-x.data)), [x], (result) {
        auto _x = result.parents[0];
        _x.grad += (result.data * (1 - result.data)) * result.grad;
    });
}

Value activateTanh(Value x) 
{
    return value(tanh(x.data), [x], (result) {
        auto _x = result.parents[0];
        _x.grad += (1 - result.data * result.data) * result.grad;
    });
}

Value activateRelu(Value x) 
{
    return value(max(0, x.data), [x], (result) {
        auto _x = result.parents[0];
        _x.grad += (result.data > 0) * result.grad;
    });
}

Value activateLeakyRelu(Value x) 
{
    return value(max(0, x.data) + 0.01 * min(0, x.data), [x], (result) {
        auto _x = result.parents[0];
        _x.grad += (result.data >= 0 ? 1 : 0.01) * result.grad;
    });
}

