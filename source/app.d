module app;

import std.stdio;
import rk.tgrad;

void main()
{
    auto a = new Value!float(3);
    auto b = new Value!float(-2);
    auto d = new Value!float(12);
    auto c = a * b;
    auto e = c + d;
    auto f = e * e;
    auto g = f / 2;

    g.backward();
    g.resetGrads();
    g.backward();

    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "a", a.data, a.grad, a.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "b",  b.data, b.grad, b.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "c",  c.data, c.grad, c.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "d",  d.data, d.grad, d.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "e",  e.data, e.grad, e.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "f",  f.data, f.grad, f.children);
    writefln("Value(label='%s', data=%s, grad=%s, children=%s)", "g",  g.data, g.grad, g.children);
}


