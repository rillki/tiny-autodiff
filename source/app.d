module app;

import std.stdio;
import rk.tgrad;

void main()
{
    auto a = value(-4);
    auto b = value(2);
    auto c = a + b;
    a.backward();
    a.writeln;
    b.writeln;
    c.writeln;
}


