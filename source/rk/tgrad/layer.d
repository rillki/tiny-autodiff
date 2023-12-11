module rk.tgrad.layer;

import rk.tgrad.value;
import rk.tgrad.activation;
import rk.tgrad.perceptron;
import std.traits : isFloatingPoint;

class Layer(size_t[2] inOutSize, T = float) if (isFloatingPoint!T) : INeuron!T
{
    Perceptron!(inOutSize[0], T)[inOutSize[1]] neurons;

    this(Value!(T) function(ref Value!T) activate)
    {
        foreach (ref n; neurons) n = new Perceptron!(inOutSize[0], T)(activate);
    }

    auto opCall(Value!(T)[] x)
    in (x.length == inOutSize[0])
    {
        Value!(T)[] result;
        foreach (ref n; neurons) result ~= n(x);
        return result;
    }

    override Value!(T)[] parameters() 
    {
        Value!(T)[] params;
        foreach (n; neurons) params ~= n.parameters();
        return params;
    }
}

unittest
{
    import std.stdio;

    auto layer = new Layer!([2, 1])((x) { return activateRelu(x); });
    auto x = [
        [value(0), value(0)],
        [value(0), value(1)],
        [value(1), value(0)],
        [value(1), value(1)],
    ];
    auto y = [
        value(0),
        value(0),
        value(0),
        value(1),
    ];
    auto output = layer(x[0]);
    output.writeln;
    layer.parameters.writeln;
}

