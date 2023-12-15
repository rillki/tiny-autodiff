module app;

import std.stdio;
import rk.tgrad;

void main()
{   
    auto neuron = new Neuron(3);
    writeln("hello");

    foreach(p; neuron.params) {
        writeln(p.data);
    }
    writeln("hello");
}

