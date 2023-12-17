module rk.tgrad.aux.math;

import rk.tgrad.core.common;
import rk.tgrad.core.value;

import std.math : log;
import std.algorithm : clamp;

enum EPSILON = 1.19209290e-7;

Value log(Value x)
{
    return value(log(x.data.clamp(EPSILON, 1 - EPSILON)), [x], (result) {
        auto _x = result.parents[0];
        _x.grad += 1/_x.data * result.grad;
    });
}

