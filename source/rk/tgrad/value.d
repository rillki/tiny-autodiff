module rk.tgrad.value;

import std.traits : isFloatingPoint;

struct Value(T) if (isFloatingPoint!T)
{
    T t;
}


