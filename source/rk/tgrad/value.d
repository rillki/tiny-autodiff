module rk.tgrad.value;

import std.traits : isFloatingPoint;

struct Value(T = float) if (isFloatingPoint!T)
{
    T data;
    T grad;
    Value!T[2] children;
}




